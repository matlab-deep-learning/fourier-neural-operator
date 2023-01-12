%% Fourier Neural Operator for 1d Burgers' Equation
% In this example we apply the <https://arxiv.org/pdf/2010.08895.pdf Fourier 
% Neural Operator> to learn the one-dimensional Burgers' equation with the following 
% definition:
% 
% $\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \frac{1}{Re}\frac{\partial^2 
% u}{\partial x^2}$,    $x \in (0,1),\space t \in (0,1]$
% 
% $u(x,0) = u_0(x) $,    $x \in (0,1)$
% 
% where $u=u\left(x,t\right)$ and $Re$ is the Reynolds number. Periodic boundary 
% conditions are imposed across the spatial domain. We learn the operator mapping 
% the initial condition $u_0$ to the solution at time $t=1$: $u_0 \longmapsto 
% u\left(x,1\right)$.
%% Data preparation
% We use the burgers_data_R10.mat, which contains initial velocities $u_0$ and 
% solutions $u\left(x,1\right)$ of the Burgers' equation which we use as training 
% inputs and targets respectively. The network inputs also consist of the spatial 
% domain $x=\left(0,1\right)$ at the desired discretization. In this example we 
% choose a grid size of $h=2^{10}$.

% Download training data.
dataDir = fullfile('data');
if ~isfolder(dataDir)
    mkdir(dataDir);
end
dataFile = fullfile(dataDir,'burgers_data_R10.mat');
if ~exist(dataFile, 'file')
    location = 'https://ssd.mathworks.com/supportfiles/nnet/data/burgers1d/burgers_data_R10.mat';
    websave(dataFile, location); 
end
data = load(dataFile, 'a', 'u');
x = data.a;
t = data.u;

% Setup.
addpath(genpath('fno'));

% Specify the number of observations in training and test data, respectively. 
numTrain = 1e3;
numTest = 1e2;

% Specify grid size and downsampling factor.
h = 2^10;
n = size(x,2);
ns = floor(n./h);

% Downsample the data for training.
xTrain = x(1:numTrain, 1:ns:n);
tTrain = t(1:numTrain, 1:ns:n);
xTest = x(end-numTest+1:end, 1:ns:n);
tTest = t(end-numTest+1:end, 1:ns:n);

% Define the grid over the spatial domain x.
xmax = 1;
xgrid = linspace(0, xmax, h);

% Combine initial velocities and spatial grid to create network
% predictors.
xTrain = cat(3, xTrain, repmat(xgrid, [numTrain 1]));
xTest = cat(3, xTest, repmat(xgrid, [numTest 1]));
%% Define network architecture
% Here we create a |dlnetwork| for the Burgers' equation problem. The network 
% accepts inputs of dimension |[h 2 miniBatchSize]|, and returns outputs of dimension 
% |[h 1 miniBatchSize]|. The network consists os multiple blocks which combine 
% spectral convolution with regular, linear convolution. The convolution in Fourier 
% space filters out higher order oscillations in the solution, while the linear 
% convolution learns local correlations.

numModes = 16;
width = 64;

lg = layerGraph([ ...
    convolution1dLayer(1, width, Name='fc0')
    
    spectralConvolution1dLayer(width, numModes, Name='specConv1')
    additionLayer(2, Name='add1')
    reluLayer(Name='relu1')
    
    spectralConvolution1dLayer(width, numModes, Name='specConv2')
    additionLayer(2, Name='add2')
    reluLayer(Name='relu2')
    
    spectralConvolution1dLayer(width, numModes, Name='specConv3')
    additionLayer(2, Name='add3')
    reluLayer(Name='relu3')
    
    spectralConvolution1dLayer(width, numModes, Name='specConv4')
    additionLayer(2, Name='add4')
    
    convolution1dLayer(1, 128, Name='fc5')
    reluLayer(Name='relu5')
    convolution1dLayer(1, 1, Name='fc6')
    ]);

lg = addLayers(lg, convolution1dLayer(1, width, Name='fc1'));
lg = connectLayers(lg, 'fc0', 'fc1');
lg = connectLayers(lg, 'fc1', 'add1/in2');

lg = addLayers(lg, convolution1dLayer(1, width, Name='fc2'));
lg = connectLayers(lg, 'relu1', 'fc2');
lg = connectLayers(lg, 'fc2', 'add2/in2');

lg = addLayers(lg, convolution1dLayer(1, width, Name='fc3'));
lg = connectLayers(lg, 'relu2', 'fc3');
lg = connectLayers(lg, 'fc3', 'add3/in2');

lg = addLayers(lg, convolution1dLayer(1, width, Name='fc4'));
lg = connectLayers(lg, 'relu3', 'fc4');
lg = connectLayers(lg, 'fc4', 'add4/in2');

numInputChannels = 2;
XInit = dlarray(ones([h numInputChannels 1]), 'SCB');
net = dlnetwork(lg, XInit);

analyzeNetwork(net)
%% Training options
% The network is trained using the standard SGDM algorithm, which 

executionEnvironment = "gpu";

batchSize = 20;
learnRate = 1e-3;
momentum = 0.9;

numEpochs = 20;
stepSize = 100;
gamma = 0.5;
expNum = 1;
checkpoint = false;
expDir = sprintf( 'checkpoints/run%g', expNum );
if ~isfolder( expDir ) && checkpoint
    mkdir(expDir)
end

vel = [];
totalIter = 0;

numTrain = size(xTrain,1);
numIterPerEpoch = floor(numTrain./batchSize);
%% Training loop
% Train the network.

if executionEnvironment == "gpu" && canUseGPU
    xTrain = gpuArray(xTrain);
    xTest = gpuArray(xTest);
end

start = tic;
figure;
clf
lineLossTrain = animatedline('Color', [0 0.4470 0.7410]);
lineLossTest = animatedline('Color', 'k', 'LineStyle', '--');
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

% Compute initial validation loss.
y = net.predict( dlarray(xTest, 'BSC') );
yTest = extractdata(permute(stripdims(y), [3 1 2]));
relLossTest = relativeL2Loss(yTest , tTest);
addpoints(lineLossTest, 0, double(relLossTest/size(xTest,1)))

% Main loop.
lossfun = dlaccelerate(@modelLoss);
for epoch = 1:numEpochs
    % Shuffle the data.
    dataIdx = randperm(numTrain);
    
    for iter = 1:numIterPerEpoch
        % Get mini-batch data.
        batchIdx = (1:batchSize) + (iter-1)*batchSize;
        idx = dataIdx(batchIdx);
        X = dlarray( xTrain(batchIdx, :, :), 'BSC' );
        T = tTrain(batchIdx, :);
        
        % Compute loss and gradients.
        [loss, dnet] = dlfeval(lossfun, X, T, net);

        % Update model parameters using SGDM update rule.
        [net, vel] = sgdmupdate(net, dnet, vel, learnRate, momentum);

        % Plot training progress.
        totalIter = totalIter + 1;
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain,totalIter,double(extractdata(loss/batchSize)))
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
        
        % Learn rate scheduling.
        if mod(totalIter, stepSize) == 0
            learnRate = gamma.*learnRate;
        end
    end
    % Compute validation loss and MSE.
    y = net.predict( dlarray(xTest, 'BSC') );
    yTest = extractdata(permute(stripdims(y), [3 1 2]));
    relLossTest = relativeL2Loss( yTest , tTest );
    mseTest = mean( (yTest(:) - tTest(:)).^2 );
    
    % Display progress.
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    numTest = size(xTest, 1);
    fprintf('Epoch = %g, train loss = %g, val loss = %g, val mse = %g, total time = %s. \n', ...
        epoch, extractdata(loss)/batchSize, relLossTest/numTest, mseTest/numTest, string(D));
    addpoints(lineLossTest, totalIter, double(relLossTest/numTest))
    
    % Checkpoints.
    if checkpoint
        filename = sprintf('checkpoints/run%g/epoch%g.mat', expNum, epoch);
        save(filename, 'net', 'epoch', 'vel', 'totalIter', 'relLossTest', 'mseTest', 'learnRate');
    end
end
%% Test on unseen, higher resolution data
% Here we take the trained network and test on unseen data with a higher spatial 
% resolution than the training data. This is an example of zero-shot super-resolution.

gridHighRes = linspace(0, xmax, n);

idxToPlot = numTrain+(1:4);
figure;
for p = 1:4
    xn = dlarray(cat(1, x(idxToPlot(p),:), gridHighRes),'CSB');
    yn = predict(net, xn);

    subplot(2, 2, p)
    plot(gridHighRes, t(idxToPlot(p),:)), hold on, plot(gridHighRes, extractdata(yn))
    axis tight
    xlabel('x')
    ylabel('U')
end
%% Helper functions

function [loss, grad] = modelLoss(x, t, net)
y = net.forward(x);
y = permute(stripdims(y), [3 1 2]);
y = stripdims(y);

loss = relativeL2Loss(y, t);

grad = dlgradient(loss, net.Learnables);
end

function loss = relativeL2Loss(y, t)
diffNorms = normFcn( (y - t) );
tNorms = normFcn( t );

loss = sum(diffNorms./tNorms, 1);
end

function n = normFcn(x)
n = sqrt( sum(x.^2, 2) );
end