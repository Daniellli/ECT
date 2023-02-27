

clc; clear; close all;
path = genpath('../../matlab');
addpath(path)



%% Evaluation on SBD
categories = categories_sbd();
% Original GT (Thin)

eval_dir = {'../../../sbd/dff/dff_val/fuse'};
result_dir = {'../../../sbd/result/evaluation/test/inst/gt_orig_thin/dff'};

t1=clock;
evaluation('../../../data/sbd-preprocess/gt_eval/gt_orig_thin/test.mat', '../../../data/sbd-preprocess/gt_eval/gt_orig_thin/inst',...
           eval_dir, result_dir, categories, 5, 99, true, 0.02)    
t2=clock;
t=etime(t2,t1);
disp(['spend time : ',num2str(t/60),' minutes'])


% Original GT (Raw)
eval_dir = {'../../../sbd/dff/dff_val/fuse'};
result_dir = {'../../../sbd/result/evaluation/test/inst/gt_orig_raw/dff'};

t1=clock;
evaluation('../../../data/sbd-preprocess/gt_eval/gt_orig_raw/test.mat', '../../../data/sbd-preprocess/gt_eval/gt_orig_raw/inst',...
           eval_dir, result_dir, categories, 5, 99, false, 0.02)

t2=clock;
t=etime(t2,t1);
disp(['spend time : ',num2str(t/60),' minutes'])