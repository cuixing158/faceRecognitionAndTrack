function  newLayerGraph = freezeWeights(lg,numsLayers)
% 功能：冻结前numsLayers层学习参数,返回新的newLayerGraph
% 输入：
%     lg， layerGraph类型，网络计算图
%     numsLayers， 1*1
%     double,前numsLayers层冻结，注意查看lg.Layers的层顺序，而非analyzeNetork()显示的顺序
% 输出：
%     newLayerGraph，layerGraph类型，冻结后的网络计算图
%
% email: cuixingxing@150@gmail.com
% 2020.11.22
%
layers = lg.Layers;
connections = lg.Connections;
for ii = 1:numsLayers
    props = properties(lg.Layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end

%% assembel
newLayerGraph = layerGraph();
for i = 1:numel(layers)
    newLayerGraph = addLayers(newLayerGraph,layers(i));
end

for c = 1:size(connections,1)
    newLayerGraph = connectLayers(newLayerGraph,connections.Source{c},connections.Destination{c});
end

end