function LP = LogPrior(hyps, GPModel)
    
    % Start log-prior
    LP = -Inf*ones(size(hyps));
    
    % Get exponential version of parameters
    thisHyps = exp(hyps);
    nFixedParams = 4; % logS, logL, logEI, logkGA, rest is noise
    
    % kernelSigma: uniform random distribution
    if isfield(GPModel.Priors, 'Sigma')
        upperLim = GPModel.Priors.Sigma(2);
        lowerLim = GPModel.Priors.Sigma(1);
    else
        upperLim = 10;
        lowerLim = 1e-10;
    end
    if thisHyps(1) <= upperLim && thisHyps(1) >= lowerLim
        LP(1) = log(1/(upperLim-lowerLim));
    end

    % kernelTheta: uniform random distribution
    if isfield(GPModel.Priors, 'Lengthscale')
        upperLim = GPModel.Priors.Lengthscale(2);
        lowerLim = GPModel.Priors.Lengthscale(1);
    else
        upperLim = 200;
        lowerLim = 1e-5;
    end
    if thisHyps(2) <= upperLim && thisHyps(2) >= lowerLim
        LP(2) = log(1/(upperLim-lowerLim));
    end

    % EI: uniform random distribution
    if isfield(GPModel.Priors, 'EI')
        upperLim = GPModel.Priors.EI(2);
        lowerLim = GPModel.Priors.EI(1);
    else
        upperLim = 1e25;
        lowerLim = eps;
    end
    if thisHyps(3) <= upperLim && thisHyps(3) >= lowerLim
        LP(3) = log(1/(upperLim-lowerLim));
    end

    % kGA: uniform random distribution
    if isfield(GPModel.Priors, 'kGA')
        upperLim = GPModel.Priors.kGA(2);
        lowerLim = GPModel.Priors.kGA(1);
    else
        upperLim = 1e25;
        lowerLim = eps;
    end
    if thisHyps(4) <= upperLim && thisHyps(4) >= lowerLim
        LP(4) = log(1/(upperLim-lowerLim));
    end

    % Noise: uniform random distribution
    if isfield(GPModel.Priors, 'Noise')
        upperLim = GPModel.Priors.Noise(2);
        lowerLim = GPModel.Priors.Noise(1);
    else
        upperLim = 1e-1;
        lowerLim = 0;
    end
    for i = nFixedParams+1:length(thisHyps)
        if thisHyps(i) <= upperLim && thisHyps(i) >= lowerLim
            LP(i) = log(1/(upperLim-lowerLim));
        end
    end    

    % Final value of log prior
    LP = sum(LP);
end