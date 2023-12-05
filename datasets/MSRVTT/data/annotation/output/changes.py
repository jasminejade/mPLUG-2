import json

# make this change in load_jsonl funciton 
# in datasets/utils.py
# returns a list
def load_jsonl(filename):
    with open(filename, 'r') as f:
        data = f.read()
    f.close()

    data_json = json.loads(data)
    # print(data_json.keys())
    return data_json['annotations']


# make this change in save_result function
# in datasets/utils.py
# returns filename
def save_result(result, result_dir, filename, is_json=True, is_list=True, remove_duplicate=""):
    if is_json:
        result_file = f'{result_dir}/{filename}3.json'
        final_result_file = f'{result_dir}/{filename}_final.json'
        result = json.dumps(result, indent=2)
        print(result)
        with open(result_file, "w") as outfile:
            outfile.write(result)
        return result_file
    else: # for list or string or whaever
        result_file = f'{result_dir}/{filename}2.json'
        final_result_file = f'{result_dir}/{filename}_final.json'
        result = json.dumps(result, indent=4)
        torch.save(result,result_file)


    if is_list:
        result = []
    else:
        result = {}

    if is_json:
        result_file = f'{result_dir}/{filename}2.json'
        res = json.load(open(result_file,'r'))
    else:
        result_file = f'{result_dir}/{filename}2.json'
        res = torch.load(result_file)            
    if is_list:
        result += res
    else:
        result.update(res)
    if remove_duplicate:
        result_new = []
        id_list = []    
        for res in result:
            if res[remove_duplicate] not in id_list:
                id_list.append(res[remove_duplicate])
                result_new.append(res)
        result = result_new  
    if is_json:                  
        json.dump(result,open(final_result_file,'w'))   
    else:            
        torch.save(result,final_result_file)     

    print('result file saved to %s'%final_result_file)   
    return final_result_file


# make this change to calc_metric function
# in video_caption_mplug2.py
def cal_metric(result_list):
    predicts = []
    answers = []
    for each in result_list:
        predicts.append(each["pred_caption"])
        answers.append(each["gold_caption"])
    evaluator = language_evaluation.CocoEvaluator(verbose=False)
    results = evaluator.run_evaluation(predicts, answers)
    print (len(result_list), results)
    return results



# make this change to evaluation function
# in video_catpiton_mplug2.py
# returns nested list of caption results
@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.to(device=device)
    model.eval()

    caption_results = []
    metrics = []
    
    print(len(data_loader))
    count = 0
    for video, video_ids, gold_caption in data_loader:
        video = video.to(device, non_blocking=True)

        topk_ids, topk_probs = model(video, train=False)
        
        result = []
        for video_id, topk_id, topk_prob, gold_caption_list in zip(video_ids, topk_ids, topk_probs, gold_caption):
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            if config["prompt"] != "":
                ans = ans.split(config["prompt"])[-1].strip()
            result.append({"video_id": video_id, "pred_caption":ans, "gold_caption": gold_caption_list})

        count += 1
        print(count, result)
        
        result_metrics = calc_metric(result)
        
        caption_results.append(result)
        metrics.append(result_metrics.item())
        
    return caption_results
# call save outside


# make this change in video_captioning_dataset class
# in dataset/video_downstream_datasets.py
# returns array, integer and list
class Dataset:
    data = 'hi'
class video_captioning_dataset(Dataset):
    def __init__(self, ann_file, transform, 
                 video_root, num_frames=16,
                 split='train', max_words=30, 
                 read_local_data=True):
        self.ann = load_jsonl(ann_file)
        self.transform = transform
        self.max_words = max_words
        self.video_root = video_root
        self.split = split
        self.num_frames = num_frames
        self.read_local_data = True
        
        def __len__(self):
            return len(self.ann)
        
        def __getitem__(self, index):
            
            ann = self.ann[index]
            
            if self.split == 'train':
                video_id = ann['image_id']
                video_path = f'{self.video_root}/{video_id}.mp4'
                while True:
                    try:
                        video_array = read_frames_decord(video_path, num_frames=self.num_frames, sample='rand')
                    except:
                        time.sleep(0.01)
                        index = 0 if index == (len(self) - 1) else index + 1
                        ann = self.ann[index]
                        video_id = ann['image_id']
                        video_path =f'{self.video_root}/{video_id}.mp4'
                        continue
                    break
                caption = pre_caption(ann['caption'], 80)
                if self.transform:
                    video_array = self.transform(video_array) # (T, C, H, W) -> (C, T, H, W)

                return video_array, caption
            
            else:
                video_id = ann['image_id']

                video_path = f'{self.video_root}/{video_id}.mp4'

                video_array = read_frames_decord(video_path, num_frames=self.num_frames, sample='middle')
                golden_captions = [x.lower() for x in ann['golden_captions']]

                if self.transform:
                    video_array = self.transform(video_array) # (T, C, H, W) -> (C, T, H, W)

                return video_array, video_id, golden_captions



def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    print("Creating video caption datasets")
    # if args.no_randaug:
    #     datasets = create_dataset('video_caption_no_randaug', config)
    # else:
    datasets = create_dataset('video_caption', config)

    # if args.distributed:
    #     num_tasks = utils.get_world_size()
    #     global_rank = utils.get_rank()
    #     samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    # else:
    #     samplers = [None, None]
    samplers = [None, None]

    train_loader, val_loader = create_loader(datasets,samplers,
                                            batch_size=[config['batch_size_train'],config['batch_size_test']],
                                            num_workers=[1, 1],is_trains=[True, False],
                                            collate_fns=[None, test_collect_fn])
    train_loader = None
    # val_loader = create_loader(datasets, config['batch_size_test'], 1, test_collect_fn)

    # tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = MPLUG2(config=config, tokenizer=tokenizer)
    model = model.to(device)

    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)

    # arg_sche = utils.AttrDict(config['schedular'])
    # train_step_per_epoch = len(train_loader)
    # print("train_step_per_epoch: {}".format(train_step_per_epoch))
    # arg_sche["num_iterations"] = max_epoch * train_step_per_epoch - arg_sche['warmup_epochs']
    # lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    # if args.do_amp:
    #     from apex import amp
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']

        # reshape positional embedding to accomodate for image resolution change
        if config["clip_name"] == "ViT-B-16":
            num_patches = int(config["image_res"] * config["image_res"]/(16*16))
        elif config["clip_name"] == "ViT-L-14":
            num_patches = int(config["image_res"] * config["image_res"]/(14*14))
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

        pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                   pos_embed.unsqueeze(0))
        state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

        if not args.evaluate:
            for key in list(state_dict.keys()):
                if ('fusion' in key or 'bert' in key) and 'decode' not in key:
                    encoder_key = key.replace('fusion.', '').replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]


        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    # if args.distributed:
    #     #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     import apex
    #     model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    #     model_without_ddp = model.module

    best_epoch = -1
    best_acc = 0
    print("Start training")
    start_time = time.time()
    data_json = load_jsonl(config["test_file"])
    caption_result, metrics = evaluation(model, val_loader, tokenizer, device, config)
    result_file = save_result(caption_result, args.result_dir, 'caption_result_zeroshot2')
    metrics_file = save_result(metrics, args.result_dir, 'metrics_result_zeroshot2')
    # if utils.is_main_process():
    #     result = cal_metric(result_file)
    #     val_stats = result
    #
    # if utils.is_main_process():
    #     log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
    #                     'epoch': -1,
    #                     }
    #     with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    #         f.write(json.dumps(log_stats) + "\n")
    #     best_acc = float(val_stats['CIDEr'])

    # dist.barrier()
    for epoch in range(start_epoch, max_epoch):
        # if epoch > 0:
        #     lr_scheduler.step(epoch + warmup_steps)

            
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)

        if args.evaluate:
            break

        caption_result, metrics = evaluation(model, val_loader, tokenizer, device, config)
        result_file = save_result(caption_result, args.result_dir, 'caption_result_epoch%d'%epoch)
        metrics_file = save_result(metrics, args.metrics_dir, 'metrics_result_epoch%d' % epoch)
        if utils.is_main_process():       
            result = cal_metric(result_file)
            val_stats = result
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                         
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))

            if float(val_stats['CIDEr']) >= best_acc:
                best_epoch = epoch
                best_acc = float(val_stats['CIDEr'])
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

        if not lr_scheduler.step_mode:
            lr_scheduler.step(epoch + warmup_steps + 1)
        dist.barrier()
        torch.cuda.empty_cache()
  
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # if utils.is_main_process():
    #     if not args.evaluate:
    #         with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
    #             f.write("best epoch: %d\n"%best_epoch)