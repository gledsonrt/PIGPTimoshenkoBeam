% System identification of Timoshenko beams via physics-informed Gaussian processes
% Tondo, G., Rau, S., Kavrakov, I., Morgenthal, G. (2023)

%% Clear workspace and prepare environment
clc; clear; close all; rng(0);
addpath(genpath('Functions'));

%% Generate analytical results
AP = struct;
AP.L = 3;
AP.x = (0:0.01:AP.L)'; 
AP.r = 1e-3; % Low rigidity: Bernoulli model
AP.EI = 11300; 
AP.kGA = 3*AP.EI/(AP.r*AP.L^2);
AP.q = -670;
AP.z = 0.03;
AP = AnalyticalBeamModel(AP);

%% Initialize GP Model
% PIGP structure
GP = struct;
GP.kernels = GetKernels();
GP.deflections = {};
GP.rotations = {};
GP.strains = {};
GP.moments = {};
GP.shears = {};
GP.loads = {};
GP.z = AP.z;
GP.jitter = 1e-15;

%% Inform boundary conditions
GP.deflections{1,1} = [0;AP.L];
GP.deflections{1,2} = [0;0];

GP.moments{1,1} = [0;AP.L];
GP.moments{1,2} = [0;0];

%% Optimise sensor placement
GP = PhysicsInfoSensorPlacement(GP, AP, {'w', 'w', 'w', 'r', 'r'});

% Plot results
figure; hold on; grid on; box on;
plot(AP.x, 0*AP.x, '-k', 'displayname', 'Beam');
plot(GP.deflections{1,1}, GP.deflections{1,2}, 'sr', 'displayname', 'BC');
plot(GP.deflections{2,1}, 0*GP.deflections{2,1}, 'ob', 'displayname', 'Deflection');
plot(GP.rotations{1,1}, 0*GP.rotations{1,1}, 'xb', 'displayname', 'Rotation');
xlabel('$x$ [m]', 'interpreter', 'latex');
legend(); xlim([-0.1, 3.1]);

%% Simulate measurements in optimal sensor locations
GP.deflections{2,2} = AP.w(any(AP.x == GP.deflections{2,1}', 2));
GP.rotations{1,2} = AP.r(any(AP.x == GP.rotations{1,1}', 2));

% Choose number of data points and simulate noise
SNR = 20;
NDP = 4;
GP.deflections{2,1} = repmat(GP.deflections{2,1}, [NDP, 1]);
GP.deflections{2,2} = repmat(GP.deflections{2,2}, [NDP, 1]);
GP.deflections{2,2} = GP.deflections{2,2} + randn(size(GP.deflections{2,2}))*max(abs(AP.w))/SNR;
GP.rotations{1,1} = repmat(GP.rotations{1,1}, [NDP, 1]);
GP.rotations{1,2} = repmat(GP.rotations{1,2}, [NDP, 1]);
GP.rotations{1,2} = GP.rotations{1,2} + randn(size(GP.rotations{1,2}))*max(abs(AP.r))/SNR;

%% Include loads into the GP
GP.loads{1,1} = (0:0.50:AP.L)';
GP.loads{1,2} = ones(size(GP.loads{1,1}))*AP.q;

%% Stiffness identification
% MCMC parameters
GP.optim.chainLength = 1e4;      
GP.optim.burnin = 2e3;           
GP.optim.thin = 2;
GP.optim.stdevSampler = 1e-2;

% Hyperparameters for optimization: start at wrong EI and kGA
GP.optim.hyps0 = log([1, 1, 0.8*AP.EI, 1.2*AP.kGA, [eps, max(abs(AP.w))/SNR, max(abs(AP.r))/SNR, eps, eps]]);

% Prior limits
GP.Priors.EI = [0.5*AP.EI, 1.5*AP.EI];
GP.Priors.kGA = [0.5*AP.kGA, 1.5*AP.kGA];

% Optimize using the Metropolis Hastings MCMC
logPosterior = @(x) LogLikelihood(x, GP) + LogPrior(x, GP);
propRnd = @(x) normrnd(x, GP.optim.stdevSampler);
[GP.optim.hypsOpt, GP.optim.likelihoodHistory, GP.optim.accRatio] = MetropolisHastings(GP.optim.hyps0, GP.optim.chainLength, logPosterior, ...
                                                                                 propRnd, [], GP.optim.burnin, GP.optim.thin, true);
                                                                             
% Plot identified stiffness
figure; 
subplot(1,2,1); hold on; grid on; box on;
histogram(exp(GP.optim.hypsOpt(:,3))./AP.EI);
xlabel('$EI/EI_{\mathrm{true}}$', 'interpreter', 'latex'); 
ylabel('Counts [-]', 'interpreter', 'latex');
subplot(1,2,2); hold on; grid on; box on;
histogram(exp(GP.optim.hypsOpt(:,4))./AP.kGA);
xlabel('$kGA/kGA_{\mathrm{true}}$', 'interpreter', 'latex'); 
ylabel('Counts [-]', 'interpreter', 'latex');

%% Make predictions                                                                    
[GP.pred.mean, GP.pred.stdev] = Predict(AP.x, GP, 100, false, true);

%% Plot predictions
figure; hold on; grid on; box on; 
fill([AP.x; flip(AP.x)], [GP.pred.mean(:,1)+GP.pred.stdev(:,1); flip(GP.pred.mean(:,1)-GP.pred.stdev(:,1))], 0.3*[1, 1, 1], ...
    'edgecolor', 'none', 'facealpha', 0.3, 'displayname', 'Pred Stdev')
plot(AP.x, AP.w, '-k', 'displayname', 'Analytical');
plot(AP.x, GP.pred.mean(:,1), '--r', 'displayname', 'Pred Mean');
plot(GP.deflections{1,1}, GP.deflections{1,2}, 'sr', 'displayname', 'BC');
plot(GP.deflections{2,1}, GP.deflections{2,2}, 'xb', 'displayname', 'Measurements');
xlabel('$x$ [m]', 'interpreter', 'latex');
ylabel('$w$ [m]', 'interpreter', 'latex');
legend(); xlim([-0.1, 3.1]);

figure; hold on; grid on; box on; 
fill([AP.x; flip(AP.x)], [GP.pred.mean(:,2)+GP.pred.stdev(:,2); flip(GP.pred.mean(:,2)-GP.pred.stdev(:,2))], 0.3*[1, 1, 1], ...
    'edgecolor', 'none', 'facealpha', 0.3, 'displayname', 'Pred Stdev')
plot(AP.x, AP.r, '-k', 'displayname', 'Analytical');
plot(AP.x, GP.pred.mean(:,2), '--r', 'displayname', 'Pred Mean');
plot(GP.rotations{1,1}, GP.rotations{1,2}, 'xb', 'displayname', 'Measurements');
xlabel('$x$ [m]', 'interpreter', 'latex');
ylabel('$\varphi$ [rad]', 'interpreter', 'latex');
legend(); xlim([-0.1, 3.1]);

figure; hold on; grid on; box on; 
fill([AP.x; flip(AP.x)], [GP.pred.mean(:,3)+GP.pred.stdev(:,3); flip(GP.pred.mean(:,3)-GP.pred.stdev(:,3))], 0.3*[1, 1, 1], ...
    'edgecolor', 'none', 'facealpha', 0.3, 'displayname', 'Pred Stdev')
plot(AP.x, AP.e, '-k', 'displayname', 'Analytical');
plot(AP.x, GP.pred.mean(:,3), '--r', 'displayname', 'Pred Mean');
xlabel('$x$ [m]', 'interpreter', 'latex');
ylabel('$\epsilon$ [-]', 'interpreter', 'latex');
legend(); xlim([-0.1, 3.1]);

figure; hold on; grid on; box on; 
fill([AP.x; flip(AP.x)], [GP.pred.mean(:,4)+GP.pred.stdev(:,4); flip(GP.pred.mean(:,4)-GP.pred.stdev(:,4))], 0.3*[1, 1, 1], ...
    'edgecolor', 'none', 'facealpha', 0.3, 'displayname', 'Pred Stdev')
plot(AP.x, AP.m, '-k', 'displayname', 'Analytical');
plot(AP.x, GP.pred.mean(:,4), '--r', 'displayname', 'Pred Mean');
xlabel('$x$ [m]', 'interpreter', 'latex');
ylabel('$M$ [Nm]', 'interpreter', 'latex');
legend(); xlim([-0.1, 3.1]);

figure; hold on; grid on; box on; 
fill([AP.x; flip(AP.x)], [GP.pred.mean(:,5)+GP.pred.stdev(:,5); flip(GP.pred.mean(:,5)-GP.pred.stdev(:,5))], 0.3*[1, 1, 1], ...
    'edgecolor', 'none', 'facealpha', 0.3, 'displayname', 'Pred Stdev')
plot(AP.x, AP.v, '-k', 'displayname', 'Analytical');
plot(AP.x, GP.pred.mean(:,5), '--r', 'displayname', 'Pred Mean');
xlabel('$x$ [m]', 'interpreter', 'latex');
ylabel('$V$ [N]', 'interpreter', 'latex');
legend(); xlim([-0.1, 3.1]);
