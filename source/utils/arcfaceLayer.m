classdef arcfaceLayer < nnet.layer.Layer
    % 功能：实质上是不带bias的“全连接层”，加入角度margin,学习每个类别的“代理特征向量”,custom ArcFace Head loss ,implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599  
    % 
    %  此层使用的方式为[...,
    %                fullyConnectedLayer(),... % 注意：此全连接层不带bias项
    %                arcfaceLayer(),...
    %                softmaxLayer(),...
    %                classificationLayer()]
    %
    % layer = arcfaceLayer();
    % validInputSize = [1,1,512,1];
    % checkLayer(layer,validInputSize,'ObservationDimension',4)
    %
    % 参考：  
    %    https://blog.csdn.net/yiran103/article/details/83684613
    %    https://github.com/cuixing158/jetson_faceTrack_pytorch
    %    offical document: Define Custom Deep Learning Layer with Learnable Parameters
    %
    % email:cuixingxing150@gmail.com
    % 2020.11.21
    %
    properties
        ClassNum
        cos_m
        sin_m
        mm
        threshold
        s
        m
    end
    properties (Learnable)
        % Layer learnable parameters
            
        % kernel parameter ,size: embedding_size*classnum, 每列代表一个“代理特征向量”
        kernel 
    end
    
    methods
        function layer = arcfaceLayer(name,embedding_size, classnum,  s, m) 
            % init function
            arguments
                name (1,1) string = "arcface_head"
                embedding_size (1,1) {mustBeNumeric}=512
                classnum (1,1) {mustBeNumeric}=4
                s (1,1) {mustBeNumeric}=64 % scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
                m (1,1) {mustBeNumeric}=0.5 % the margin value, default is 0.5
            end
 
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "ArcFace with " + string(embedding_size) + " embedding size"+" classnum:"+string(classnum)+" scalar:"+string(s)+" m:"+string(m);
        
            % 
            layer.ClassNum = classnum;
            
            % Initialize kernel coefficient, Glorot Initialization(also
            % known as Xavier),"Initialize Learnable Parameters for
            % Model Functions"
            Z = 2*rand(embedding_size,classnum,'single')-1;
            bound = sqrt(6 / (embedding_size + classnum));
            weights = bound * Z;
            layer.kernel = dlarray(weights);
            
            % set layer intermediate parameters
            layer.cos_m = cos(m);
            layer.sin_m = sin(m);
            layer.mm = m.*sin(m);
            layer.threshold = cos(pi-m);
            layer.s = s;
            layer.m = m;
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            % 
            % 输入：
            %     X ，必须为 1*1*embedding_size*bs四维矩阵，前一层接fullyconnected，
            %     输入形式h*w*c*bs
            % 输出：
            %     Z， 必须为1*1*classnum*bs四维矩阵，后一层接softmax，输入形式h*w*c*bs
            %
            % 注意：该函数中每个operation要支持dlarray操作，参考官方文档“List of Functions with
            % dlarray Support”查阅
            %

            X = squeeze(X);% embedding_size*bs
            X = X'; % bs*embedding_size
            X = X./sqrt(sum(X.^2,2));
            kernel_norm = layer.kernel./sqrt(sum(layer.kernel.^2,1)); % embedding_size*classnum
            
            % cos(theta+m)
            cos_theta = X*kernel_norm; % bs*classnum
            cos_theta_2 = cos_theta.^2;
            sin_theta_2 = 1 - cos_theta_2;
            sin_theta = sqrt(sin_theta_2);
            cos_theta_m = (cos_theta * layer.cos_m - sin_theta * layer.sin_m); %  bs*classnum,数学公式, cos(alpha+beta) = cos(alpha)*cos(beta)-sin(alpha)*sin(beta)
            % this condition controls the theta+m should in range [0, pi]
            %      0<=theta+m<=pi
            %   -m<=theta<=pi-m
            cond_v = cos_theta - layer.threshold; % bs*classnum, 按道理此值大于0
            cond_mask = cond_v <= 0; % bs*classnum, 找出小于0的索引
            keep_val = cos_theta - layer.mm; % when theta not in [0,pi], use cosface instead,# bs*classnum
            cos_theta_m(cond_mask) = keep_val(cond_mask); % bs*classnum
            output = cos_theta_m;

            Z = layer.s.*output; % scale up in order to make softmax work, first introduced in normface  ,bs*classnum
            Z = Z'; % classnum*bs
            Z = reshape(Z,1,1,size(Z,1),size(Z,2));% 1*1*classnum*bs
        end
    end
end

