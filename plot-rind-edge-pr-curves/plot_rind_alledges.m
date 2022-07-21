function b = plot_ED()


clear;

colors = {
    [1.0000, 0.0000, 0.0000]
    [0.7059, 0.5333, 0.8824]
    [0.8000, 0.8000, 0.1000]
    [0.9373, 0.6863, 0.1255]
    [0.0588, 0.6471, 0.6471]
    [0.0000, 1.0000, 1.0000]
    [0.0000, 0.0000, 1.0000]
    [0.7098, 0.2000, 0.3608]
    [0.7176, 0.5137, 0.4392]
    [0.4157, 0.5373, 0.0824]
    [0.5490, 0.5490, 0.4549]
    [0.5490, 0.6490, 0.4549]
};

lines = {'-','-','-','-','-','-','--','--','--','--','--','--'};

names = {
    'EdgeCerberus'
    'RINDNet'
    'DFF'
    'RCF'
    'CASENet'
    'BDCN'
    'OFNet'
    'DOOBNet'
    'DeepLabV3+'
    'HED'
    'CED'
    'DexiNed'
};

years = {
    ' (2022)'
    ' (2021)'
    ' (2019)'
    ' (2017)'
    ' (2017)'
    ' (2019)'
    ' (2019)'
    ' (2018)'
    ' (2018)'
    ' (2015)'
    ' (2017)'
    ' (2020)'
};

edgesEvalPlot('eval_rind_edges', names, colors, lines, years, true);
saveas(gcf,'edge_detection.png');
close all;
b="hello world"

end