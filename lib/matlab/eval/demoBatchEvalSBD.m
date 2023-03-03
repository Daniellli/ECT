

function results = demoBatchEvalSBD(eval_dir,result_dir)
%clc; clear; close all;
path = genpath('../../matlab');
addpath(path)



%% Evaluation on SBD
categories = categories_sbd();
% Original GT (Thin)

%eval_dir = {'../../../sbd/dff/dff_val/fuse'};
%result_dir = {'../../../sbd/result/evaluation/test/inst/gt_orig_thin/dff'};

t1=clock;
%eval_resuls = evaluation('../../../data/sbd-preprocess/gt_eval/gt_orig_thin/test.mat',
%                '../../../data/sbd-preprocess/gt_eval/gt_orig_thin/inst',...
%                eval_dir, result_dir, categories, 5, 99, true, 0.02); 


eval_resuls = evaluation('/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/sbd-preprocess/gt_eval/gt_orig_thin/test.mat',...
                '/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/sbd-preprocess/gt_eval/gt_orig_thin/inst',...
                eval_dir, result_dir, categories, 5, 99, true, 0.02); 


results = struct();
num_cls = length(categories);
for idx_cls = 1:num_cls
    new_id = matlab.lang.makeValidName(categories{idx_cls});
    results.(new_id)=eval_resuls(idx_cls);
    fprintf('%2d %14s:  %.3f\n', idx_cls, categories{idx_cls}, results.(new_id));
end
fprintf('\n      Mean MF-ODS:  %.3f\n\n', mean(eval_resuls));
results.('meanMF_ODS')=mean(eval_resuls);
%disp(results);


t2=clock;
t=etime(t2,t1);
disp(['spend time : ',num2str(t/60),' minutes'])





% Original GT (Raw)
%eval_dir = {'../../../sbd/dff/dff_val/fuse'};
%result_dir = {'../../../sbd/result/evaluation/test/inst/gt_orig_raw/dff'};
%evaluation('../../../data/sbd-preprocess/gt_eval/gt_orig_raw/test.mat', '../../../data/sbd-preprocess/gt_eval/gt_orig_raw/inst',...
%           eval_dir, result_dir, categories, 5, 99, false, 0.02)
%disp(['spend time : ',num2str(t/60),' minutes'])