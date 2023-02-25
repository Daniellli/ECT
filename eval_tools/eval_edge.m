

function  eval_results = eval_edge(eval_data_dir,test_list)

% Command to run.
% (echo "data_dir = '../output/epoch-x-test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
% Data directory data_dir should be defined outside.
clc
% 下面这行会导致传入的变量eval_data_dir也清空, 所以需要注释掉
%clear  
%test_list={'depth','normal','reflectance','illumination'};
disp(test_list);
% data_root='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/model_res'; 
disp(eval_data_dir)
data_root=eval_data_dir;
%============ success to import pdollar_toolbox
addpath(genpath("/home/DISCOVER_summer2022/xusc/matlab/pdollar_toolbox"));savepath;
%============
eval_results=rand(length(test_list),4); % 每行4个value

for test_index=1:size(test_list,2)
    test_type=test_list{test_index};
    data_dir = [data_root,'/',test_type];
    fprintf('Data dir: %s\n', data_dir);
    addpath(genpath('./edges'));
    addpath(genpath('./toolbox.badacost.public'));
    
    % Section 1: NMS process (formerly nms_process.m from HED repo).
    disp('NMS process...')
    mat_dir = fullfile(data_dir, 'met');
    nms_dir = fullfile(data_dir, 'nms');
    mkdir(nms_dir)
    
    files = dir(mat_dir+"/*.mat");
    % files = files(3:end,:);  % It means all files except ./.. are considered.

    mat_names = cell(1,size(files, 1));
    nms_names = cell(1,size(files, 1));
    for i = 1:size(files, 1)
        mat_names{i} = files(i).name;    
        nms_names{i} = [files(i).name(1:end-4), '.png']; % Output PNG files.
    end
    
    for i = 1:size(mat_names,2)

        matObj = matfile(fullfile(mat_dir, mat_names{i})); % Read MAT files.
        varlist = who(matObj);
        
        x = matObj.(char(varlist));
        E=convTri(single(x),1);
        [Ox,Oy]=gradient2(convTri(E,4));
        [Oxx,~]=gradient2(Ox); 
        [Oxy,Oyy]=gradient2(Oy);
        O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
        E=edgesNmsMex(E,O,1,5,1.01,4);
        %  RCF explaination for  NMS as listed as follows : 
        % 2 for BSDS500 and Multi-cue datasets, 4 for NYUD dataset 
        % E = edgesNmsMex(E, O, 2, 5, 1.01, 4);
        % edgesNmsMex(edge , O, r, s, m, nThreads);
        %E=edgesNmsMex(E,O,2,5,1.01,4);
        
        imwrite(uint8(E*255),fullfile(nms_dir, nms_names{i}))
    end
    
    % Section 2: Evaluate the edges (formerly EvalEdge.m from HED repo).
    disp('Evaluate the edges...');

    
    %gtDir  = ['/home/DISCOVER_summer2022/xusc/exp/data/BSDS-RIND/testgt/',test_type];
    
    % for SBU shadow edge detection 
    %gtDir  = ['/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/SBU/SBU-shadow/SBU-Test/EdgeMapMat'];

    % for ISTD shadow edge detection 
    %gtDir  = ['/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/ISTD/ISTD_Dataset/test/EdgeMapMat'];

    % for NYUD2 normal and depth  edge detection 
    %gtDir  = ['/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/nyud2/NYU_origin/depth_normal_edges_canny/threashold_decay/','nyu_',test_type,'_edges_crop_mat'];
    %gtDir  = ['/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/nyud2/NYU_origin/depth_normal_edges_3x3/threashold_decay/','nyu_',test_type,'_edges_crop_mat'];
    gtDir  = ['/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/nyud2/NYU_origin/depth_normal_edges_3x3/tmp/','nyu_',test_type,'_edges_crop_mat'];
    


    resDir = fullfile(data_dir, 'nms');
    edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);
    figure; 
    eval_res = edgesEvalPlot(resDir,['modelname-',test_type]);
    saveas(gcf,[data_dir,'/modelname-',test_type,'.jpg']);
    %===============
    eval_results(test_index,:) = eval_res;
    %===============
    close all;
end
end

