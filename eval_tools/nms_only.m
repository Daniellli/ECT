function holder = nms_only(eval_data_dir,test_list)
clc
disp(test_list);
disp(eval_data_dir)
data_root=eval_data_dir;
addpath(genpath("/home/DISCOVER_summer2022/xusc/matlab/pdollar_toolbox"));savepath;
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
    
    files = dir(mat_dir);
    files = files(3:end,:);  % It means all files except ./.. are considered.
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
        imwrite(uint8(E*255),fullfile(nms_dir, nms_names{i}))
    end

end

holder='hello world';

end