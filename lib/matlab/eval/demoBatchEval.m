clc; clear; close all;
path = genpath('../../matlab');
addpath(path)

%% Evaluation on Cityscapes
categories = categories_city();
% Original GT (Thin)
eval_dir = {'../../../cityscapes/dff/dff_val/fuse'};
result_dir = {'../../../cityscapes/result/evaluation/test/inst/gt_orig_thin/dff'};
evaluation('../../../data/cityscapes-preprocess/gt_eval/gt_thin/val.mat', '../../../data/cityscapes-preprocess/gt_eval/gt_thin',...
           eval_dir, result_dir, categories, 0, 99, true, 0.02) % 0.0035
           
% Original GT (Raw)
%eval_dir = {'../../../cityscapes/dff/dff_val/fuse'};
%result_dir = {'../../../cityscapes/result/evaluation/test/inst/gt_orig_raw/dff'};
%evaluation('../../../data/cityscapes-preprocess/gt_eval/gt_raw/val.mat', '../../../data/cityscapes-preprocess/gt_eval/gt_raw',...
%           eval_dir, result_dir, categories, 0, 99, false, 0.02) % 0.0035
