


function [eval_resuls] = demoBatchEvalCityscapes()

clc; clear; close all;
path = genpath('../../matlab');
addpath(path)

%% Evaluation on Cityscapes
categories = categories_city();
% Original GT (Thin)

model_name='cerberus';

eval_dir = {['../../../cityscapes/',model_name,'/',model_name,'_val/fuse']};
result_dir = {['../../../cityscapes/result/',model_name,'/evaluation/test/inst/gt_orig_thin/dff']};
t1=clock;
eval_resuls = evaluation('../../../data/cityscapes-preprocess/gt_eval/gt_thin/val.mat', '../../../data/cityscapes-preprocess/gt_eval/gt_thin/inst',...
           eval_dir, result_dir, categories, 0, 99, true, 0.02) % 0.0035


results = struct();
num_cls = length(categories);
for idx_cls = 1:num_cls
    results.(categories{idx_cls})=result_f(idx_cls);
    fprintf('%2d %14s:  %.3f\n', idx_cls, categories{idx_cls}, result_f(idx_cls));
end

disp(results)




t2=clock;
t=etime(t2,t1);
disp(['spend time : ',num2str(t/60),' minutes'])

% Original GT (Raw)
%eval_dir = {'../../../cityscapes/dff/dff_val/fuse'};
%result_dir = {'../../../cityscapes/result/evaluation/test/inst/gt_orig_raw/dff'};
%evaluation('../../../data/cityscapes-preprocess/gt_eval/gt_raw/val.mat', '../../../data/cityscapes-preprocess/gt_eval/gt_raw/inst',...
%           eval_dir, result_dir, categories, 0, 99, false, 0.02) % 0.0035



end

