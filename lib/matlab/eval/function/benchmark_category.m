function result_cls = benchmark_category(file_list, result_dir, gt_dir, idx_cls, margin, nthresh, thinpb, maxDist)



% get the dataset
tmp = strsplit(gt_dir,'/');
tmp = strsplit(string(tmp(end-3)),'-');
dataset = tmp(1);

if strcmp(dataset,'sbd')
    disp('evaluate on SBD');
    result_img = evaluate_imgs_sbd(file_list, result_dir, gt_dir, idx_cls, margin, nthresh, thinpb, maxDist);
else
    disp('evaluate on cityscapes');
    result_img = evaluate_imgs(file_list, result_dir, gt_dir, idx_cls, margin, nthresh, thinpb, maxDist);
end;



result_cls = collect_eval_bdry(result_img);