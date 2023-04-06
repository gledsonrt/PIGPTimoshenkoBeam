function [dataMeans, dataStdev] = Predict(xStar, GPModel, NMC, returnKernel, verbose)
% probInference - makes inferences on the model basedn on Bayesian
% estimations of the hyperparameters. This includes model uncertainty on
% the parameters themselves.

    % Check if kernel should be returned
    if nargin < 4; returnKernel = false; end
    if nargin < 5; verbose = false; end
    % Check if it is parameters are probabilistic
    isProb = false;
    hyps = GPModel.optim.hypsOpt;
    if any(size(hyps) == 1)
        if ~returnKernel && verbose
            fprintf('Running inference... ')
        end
        hypModels = hyps; NMC = 1;
    else
        % Allocated vector for samples of parameters
        isProb = true;
        NMC = min([NMC, size(hyps, 1)]);
        idx = randi(size(hyps,1), [NMC,1]);
        hypModels = hyps(idx,:);        
        if ~returnKernel && verbose
            fprintf(1,'Running probabilistic inference: %3d%%\n', 0);
        end
    end
    
    % Storage vector for means and covariances
    % Size: xStar, (1, xStar) x type x N
    means = zeros(length(xStar), 1, 6, NMC);
    covars = zeros(length(xStar), length(xStar), 6, NMC);
    
    % Plot types
    quantities = {'deflections','rotations','strains','moments','shears','loads'};
    
    % Loop over MC cases
    mcN = 0;
    while mcN < NMC
%         try
            % Select parameters
            hyps = hypModels(mcN+1,:);
            % Unpack kernel and physical parameters
            logsigma = hyps(1);
            logtheta = hyps(2);
            EI = hyps(3);
            kGA = hyps(4);

            % Create output target vector
            y = [];
            n = [];
            for i = 1:size(GPModel.deflections,1)
                y = cat(1, y, GPModel.deflections{i,2});
                n = cat(1, n, length(GPModel.deflections{i,1}));
            end
            for i = 1:size(GPModel.rotations,1)
                y = cat(1, y, GPModel.rotations{i,2});
                n = cat(1, n, length(GPModel.rotations{i,1}));
            end
            for i = 1:size(GPModel.strains,1)
                y = cat(1, y, GPModel.strains{i,2});
                n = cat(1, n, length(GPModel.strains{i,1}));
            end
            for i = 1:size(GPModel.moments,1)
                y = cat(1, y, GPModel.moments{i,2});
                n = cat(1, n, length(GPModel.moments{i,1}));
            end
            for i = 1:size(GPModel.shears,1)
                y = cat(1, y, GPModel.shears{i,2});
                n = cat(1, n, length(GPModel.shears{i,1}));
            end
            for i = 1:size(GPModel.loads,1)
                y = cat(1, y, GPModel.loads{i,2});
                n = cat(1, n, length(GPModel.loads{i,1}));
            end

            % Block covariance matrix positions
            nB = cumsum(n);
            nB = [1; nB(1:end-1)+1];
            nB = cat(2, nB, cumsum(n));

            % Recover kernel
            K = LogLikelihood(hyps, GPModel, true);
            if isempty(K); break; end

            % Cholesky
            [L,p] = chol(K,'lower');         
            % if p > 0; error('Numerical issues on covariance.'); end
            while p>0
                K = K + eye(length(K))*GPModel.jitter*10; 
                [L,p] = chol(K,'lower') ;
            end

            % Loop over inference types
            Qstore = [];
            for t = 1:length(quantities)
                quantity = quantities{t};
                % Q-Matrix
                Q = zeros(length(xStar),size(L,1));
                cC = 1; % Kernel block-column
                switch quantity
                    case 'deflections'
                        for i = 1:size(GPModel.deflections,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kww(EI, kGA, logsigma, logtheta, xStar, GPModel.deflections{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.rotations,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kwr(EI, kGA, logsigma, logtheta, xStar, GPModel.rotations{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.strains,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kwe(EI, kGA, logsigma, logtheta, xStar, GPModel.strains{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.moments,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kwm(EI, kGA, logsigma, logtheta, xStar, GPModel.moments{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.shears,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kwv(EI, kGA, logsigma, logtheta, xStar, GPModel.shears{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.loads,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kwq(EI, kGA, logsigma, logtheta, xStar, GPModel.loads{i,1}');
                            cC = cC + 1;
                        end
                    case 'rotations'
                        for i = 1:size(GPModel.deflections,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.krw(EI, kGA, logsigma, logtheta, xStar, GPModel.deflections{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.rotations,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.krr(EI, kGA, logsigma, logtheta, xStar, GPModel.rotations{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.strains,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kre(EI, kGA, logsigma, logtheta, xStar, GPModel.strains{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.moments,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.krm(EI, kGA, logsigma, logtheta, xStar, GPModel.moments{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.shears,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.krv(EI, kGA, logsigma, logtheta, xStar, GPModel.shears{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.loads,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.krq(EI, kGA, logsigma, logtheta, xStar, GPModel.loads{i,1}');
                            cC = cC + 1;
                        end
                    case 'strains'
                        for i = 1:size(GPModel.deflections,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kew(EI, kGA, logsigma, logtheta, xStar, GPModel.deflections{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.rotations,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.ker(EI, kGA, logsigma, logtheta, xStar, GPModel.rotations{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.strains,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kee(EI, kGA, logsigma, logtheta, xStar, GPModel.strains{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.moments,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kem(EI, kGA, logsigma, logtheta, xStar, GPModel.moments{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.shears,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kev(EI, kGA, logsigma, logtheta, xStar, GPModel.shears{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.loads,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.keq(EI, kGA, logsigma, logtheta, xStar, GPModel.loads{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                    case 'moments'
                        for i = 1:size(GPModel.deflections,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kmw(EI, kGA, logsigma, logtheta, xStar, GPModel.deflections{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.rotations,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kmr(EI, kGA, logsigma, logtheta, xStar, GPModel.rotations{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.strains,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kme(EI, kGA, logsigma, logtheta, xStar, GPModel.strains{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.moments,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kmm(EI, logsigma, logtheta, xStar, GPModel.moments{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.shears,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kmv(EI, logsigma, logtheta, xStar, GPModel.shears{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.loads,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kmq(EI, logsigma, logtheta, xStar, GPModel.loads{i,1}');
                            cC = cC + 1;
                        end
                    case 'shears'
                        for i = 1:size(GPModel.deflections,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kvw(EI, kGA, logsigma, logtheta, xStar, GPModel.deflections{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.rotations,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kvr(EI, kGA, logsigma, logtheta, xStar, GPModel.rotations{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.strains,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kve(EI, kGA, logsigma, logtheta, xStar, GPModel.strains{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.moments,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kvm(EI, logsigma, logtheta, xStar, GPModel.moments{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.shears,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kvv(EI, logsigma, logtheta, xStar, GPModel.shears{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.loads,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kvq(EI, logsigma, logtheta, xStar, GPModel.loads{i,1}');
                            cC = cC + 1;
                        end
                    case 'loads'
                        for i = 1:size(GPModel.deflections,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kqw(EI, kGA, logsigma, logtheta, xStar, GPModel.deflections{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.rotations,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kqr(EI, kGA, logsigma, logtheta, xStar, GPModel.rotations{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.strains,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kqe(EI, kGA, logsigma, logtheta, xStar, GPModel.strains{i,1}', GPModel.z);
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.moments,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kqm(EI, logsigma, logtheta, xStar, GPModel.moments{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.shears,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kqv(EI, logsigma, logtheta, xStar, GPModel.shears{i,1}');
                            cC = cC + 1;
                        end
                        for i = 1:size(GPModel.loads,1)
                            Q(:,nB(cC,1):nB(cC,2)) = GPModel.kernels.kqq(EI, logsigma, logtheta, xStar, GPModel.loads{i,1}');
                            cC = cC + 1;
                        end
                    otherwise
                        error('Wrong inference quantity. Valid types are: "deflections", "rotations", "strains", "moments", "shears" and "loads".')
                end
                
                % Prediction of mean and covariance
                Qstore = cat(1, Qstore, Q);
                pred = Q*(L'\(L\y)); 
                switch quantity
                    case 'deflections'
                        covar = GPModel.kernels.kww(EI, kGA, logsigma,logtheta,xStar,xStar') - Q*(L'\(L\Q'));
                    case 'rotations'
                        covar = GPModel.kernels.krr(EI, kGA, logsigma,logtheta,xStar,xStar') - Q*(L'\(L\Q'));
                    case 'strains'
                        covar = GPModel.kernels.kee(EI, kGA, logsigma,logtheta,xStar,xStar',GPModel.z) - Q*(L'\(L\Q'));
                    case 'moments'
                        covar = GPModel.kernels.kmm(EI,logsigma,logtheta,xStar,xStar') - Q*(L'\(L\Q'));
                    case 'shears'
                        covar = GPModel.kernels.kvv(EI,logsigma,logtheta,xStar,xStar') - Q*(L'\(L\Q'));
                    case 'loads'
                        covar = GPModel.kernels.kqq(EI,logsigma,logtheta,xStar,xStar') - Q*(L'\(L\Q'));
                end

                % Store means and covariances
                means(:,1,t,mcN+1) = pred;
                covars(:,:,t,mcN+1) = covar;

            end
            if isProb && ~returnKernel && verbose
                fprintf(1,'\b\b\b\b\b%3.0f%%\n', 100*((mcN+1)/NMC));
            end
            mcN = mcN + 1;
    end
    
    if isempty(K)
        means = zeros(length(xStar), 1, 6, NMC);
        covars(:,:,1,1) = GPModel.kernels.kww(EI, kGA, logsigma,logtheta,xStar,xStar');  
        covars(:,:,2,1) = GPModel.kernels.krr(EI, kGA, logsigma,logtheta,xStar,xStar');    
        covars(:,:,3,1) = GPModel.kernels.kee(EI, kGA, logsigma,logtheta,xStar,xStar',GPModel.z);    
        covars(:,:,4,1) = GPModel.kernels.kmm(EI, logsigma,logtheta,xStar,xStar');    
        covars(:,:,5,1) = GPModel.kernels.kvv(EI, logsigma,logtheta,xStar,xStar'); 
        covars(:,:,6,1) = GPModel.kernels.kqq(EI, logsigma,logtheta,xStar,xStar');    
    end
    
    % Finalize covariance calculation
    if size(hypModels, 1) == 1
        dataMeans = squeeze(mean(means, 4));
        if ~returnKernel
            dataStdev = zeros(length(xStar), 6);
            for t = 1:length(quantities)
                dataStdev(:,t) = sqrt(abs(diag(covars(:,:,t))));
            end
        else
            dataStdev = zeros(length(xStar), length(xStar), 6);
            for t = 1:length(quantities)
                dataStdev(:,:,t) = covars(:,:,t);
            end
        end
    else
        dataMeans = squeeze(mean(means, 4));
        addCovars = zeros(size(covars));
        for mcN = 1:NMC
            for t = 1:length(quantities)
                deltaMean = squeeze(means(:,1,t,mcN)) - dataMeans(:,t);
                addCovars(:,:,t,mcN) = deltaMean*deltaMean';
            end
        end
        dataCovar = mean(covars, 4) + mean(addCovars,4);
        dataStdev = zeros(size(dataMeans));
        for i = 1:length(quantities)
            dataStdev(:,i) = sqrt(abs(diag(dataCovar(:,:,i))));
        end
    end
    
    if ~isProb && ~returnKernel && verbose
        fprintf('Done!\n')
    end
    
end