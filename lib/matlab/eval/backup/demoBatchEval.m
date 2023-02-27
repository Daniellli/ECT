clc; clear; close all;
path = genpath('../../matlab');
addpath(path)


% about 44 minites  with 6 workers
% about 11 minites  with 128 workers

%% Evaluation on Cityscapes
categories = categories_city();


t1=clock;
% Original GT (Thin)
eval_dir = {'../../../cityscapes/dff/dff_val/fuse'};
result_dir = {'../../../cityscapes/result/evaluation/test/inst/gt_orig_thin/dff'};
evaluation('../../../data/cityscapes-preprocess/gt_eval/gt_thin/val.mat', '../../../data/cityscapes-preprocess/gt_eval/gt_thin/inst',...
           eval_dir, result_dir, categories, 0, 99, true, 0.02) % 0.0035

t2=clock;
t=etime(t2,t1);
disp(['spend time : ',num2str(t/60),' minutes'])




t1=clock;
% Original GT (Raw)
eval_dir = {'../../../cityscapes/dff/dff_val/fuse'};
result_dir = {'../../../cityscapes/result/evaluation/test/inst/gt_orig_raw/dff'};
evaluation('../../../data/cityscapes-preprocess/gt_eval/gt_raw/val.mat', '../../../data/cityscapes-preprocess/gt_eval/gt_raw/inst',...
           eval_dir, result_dir, categories, 0, 99, false, 0.02) % 0.0035

t2=clock;
t=etime(t2,t1);
disp(['spend time : ',num2str(t/60),' minutes'])