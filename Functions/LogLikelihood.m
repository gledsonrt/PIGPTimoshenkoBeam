function LOGLIK = LogLikelihood(hyps, GPModel, returnKern) 
    % Check if we should return the kernel or the loglik
    if nargin == 2; returnKern = false; end
    
    % Start algorithm
    % Unpack kernel and physical parameters
    nFixedParams = 4;
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

    % Initialize covariance matrix
    K = zeros(sum(n));

    % Kernel block-line
    cL = 1; 

    % Build deflection parts    
    for i = 1:size(GPModel.deflections,1)
        cC = 1; % Kernel block-column
        for j = 1:size(GPModel.deflections,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kww(EI, kGA, logsigma, logtheta, GPModel.deflections{i,1}, GPModel.deflections{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.rotations,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kwr(EI, kGA, logsigma, logtheta, GPModel.deflections{i,1}, GPModel.rotations{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.strains,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kws(EI, kGA, logsigma, logtheta, GPModel.deflections{i,1}, GPModel.strains{j,1}', GPModel.z);
            cC = cC + 1;
        end
        for j = 1:size(GPModel.moments,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kwm(EI, kGA, logsigma, logtheta, GPModel.deflections{i,1}, GPModel.moments{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.shears,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kwv(EI, kGA, logsigma, logtheta, GPModel.deflections{i,1}, GPModel.shears{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.loads,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kwq(EI, kGA, logsigma, logtheta, GPModel.deflections{i,1}, GPModel.loads{j,1}');
            cC = cC + 1;
        end
        cL = cL + 1;
    end

    % Build rotation parts    
    for i = 1:size(GPModel.rotations,1)
        cC = 1; % Kernel block-column
        for j = 1:size(GPModel.deflections,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.krw(EI, kGA, logsigma, logtheta, GPModel.rotations{i,1}, GPModel.deflections{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.rotations,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.krr(EI, kGA, logsigma, logtheta, GPModel.rotations{i,1}, GPModel.rotations{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.strains,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.krs(EI, kGA, logsigma, logtheta, GPModel.rotations{i,1}, GPModel.strains{j,1}', GPModel.z);
            cC = cC + 1;
        end
        for j = 1:size(GPModel.moments,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.krm(EI, kGA, logsigma, logtheta, GPModel.rotations{i,1}, GPModel.moments{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.shears,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.krv(EI, kGA, logsigma, logtheta, GPModel.rotations{i,1}, GPModel.shears{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.loads,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.krq(EI, kGA, logsigma, logtheta, GPModel.rotations{i,1}, GPModel.loads{j,1}');
            cC = cC + 1;
        end
        cL = cL + 1;
    end

    % Build Strain parts    
    for i = 1:size(GPModel.strains,1)
        cC = 1; % Kernel block-column
        for j = 1:size(GPModel.deflections,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.ksw(EI, kGA, logsigma, logtheta, GPModel.strains{i,1}, GPModel.deflections{j,1}', GPModel.z);
            cC = cC + 1;
        end
        for j = 1:size(GPModel.rotations,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.ksr(EI, kGA, logsigma, logtheta, GPModel.strains{i,1}, GPModel.rotations{j,1}', GPModel.z);
            cC = cC + 1;
        end
        for j = 1:size(GPModel.strains,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kss(logsigma, logtheta, GPModel.strains{i,1}, GPModel.strains{j,1}', GPModel.z);
            cC = cC + 1;
        end
        for j = 1:size(GPModel.moments,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.ksm(EI, logsigma, logtheta, GPModel.strains{i,1}, GPModel.moments{j,1}', GPModel.z);
            cC = cC + 1;
        end
        for j = 1:size(GPModel.shears,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.ksv(EI, logsigma, logtheta, GPModel.strains{i,1}, GPModel.shears{j,1}', GPModel.z);
            cC = cC + 1;
        end
        for j = 1:size(GPModel.loads,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.ksq(EI, logsigma, logtheta, GPModel.strains{i,1}, GPModel.loads{j,1}', GPModel.z);
            cC = cC + 1;
        end
        cL = cL + 1;
    end

    % Build moment parts    
    for i = 1:size(GPModel.moments,1)
        cC = 1; % Kernel block-column
        for j = 1:size(GPModel.deflections,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kmw(EI, kGA, logsigma, logtheta, GPModel.moments{i,1}, GPModel.deflections{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.rotations,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kmr(EI, kGA, logsigma, logtheta, GPModel.moments{i,1}, GPModel.rotations{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.strains,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kms(EI, logsigma, logtheta, GPModel.moments{i,1}, GPModel.strains{j,1}', GPModel.z);
            cC = cC + 1;
        end
        for j = 1:size(GPModel.moments,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kmm(EI, logsigma, logtheta, GPModel.moments{i,1}, GPModel.moments{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.shears,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kmv(EI, logsigma, logtheta, GPModel.moments{i,1}, GPModel.shears{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.loads,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kmq(EI, logsigma, logtheta, GPModel.moments{i,1}, GPModel.loads{j,1}');
            cC = cC + 1;
        end
        cL = cL + 1;
    end

    % Build shear parts    
    for i = 1:size(GPModel.shears,1)
        cC = 1; % Kernel block-column
        for j = 1:size(GPModel.deflections,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kvw(EI, kGA, logsigma, logtheta, GPModel.shears{i,1}, GPModel.deflections{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.rotations,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kvr(EI, kGA, logsigma, logtheta, GPModel.shears{i,1}, GPModel.rotations{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.strains,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kvs(EI, logsigma, logtheta, GPModel.shears{i,1}, GPModel.strains{j,1}', GPModel.z);
            cC = cC + 1;
        end
        for j = 1:size(GPModel.moments,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kvm(EI, logsigma, logtheta, GPModel.shears{i,1}, GPModel.moments{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.shears,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kvv(EI, logsigma, logtheta, GPModel.shears{i,1}, GPModel.shears{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.loads,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kvq(EI, logsigma, logtheta, GPModel.shears{i,1}, GPModel.loads{j,1}');
            cC = cC + 1;
        end
        cL = cL + 1;
    end

    % Build load parts    
    for i = 1:size(GPModel.loads,1)
        cC = 1; % Kernel block-column
        for j = 1:size(GPModel.deflections,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kqw(EI, kGA, logsigma, logtheta, GPModel.loads{i,1}, GPModel.deflections{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.rotations,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kqr(EI, kGA, logsigma, logtheta, GPModel.loads{i,1}, GPModel.rotations{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.strains,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kqs(EI, logsigma, logtheta, GPModel.loads{i,1}, GPModel.strains{j,1}', GPModel.z);
            cC = cC + 1;
        end
        for j = 1:size(GPModel.moments,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kqm(EI, logsigma, logtheta, GPModel.loads{i,1}, GPModel.moments{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.shears,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kqv(EI, logsigma, logtheta, GPModel.loads{i,1}, GPModel.shears{j,1}');
            cC = cC + 1;
        end
        for j = 1:size(GPModel.loads,1)
            K(nB(cL,1):nB(cL,2),nB(cC,1):nB(cC,2)) = GPModel.kernels.kqq(EI, logsigma, logtheta, GPModel.loads{i,1}, GPModel.loads{j,1}');
            cC = cC + 1;
        end
        cL = cL + 1;
    end

    % Gaussian noise model
    for i = 1:size(nB,1)
        if isempty(K); break; end
        K(nB(i,1):nB(i,2),nB(i,1):nB(i,2)) = K(nB(i,1):nB(i,2),nB(i,1):nB(i,2)) + eye(n(i))*exp(hyps(i+nFixedParams))^2;
    end

    % Assess numerical errors
    numStabil = GPModel.jitter;
    K = K + eye(sum(n))*numStabil;

    % Return kernel or move on?
    if returnKern
        LOGLIK = K;
        return
    end

    % Cholesky
    [L,p] = chol(K,'lower'); 
    if p > 0
        LOGLIK = -Inf; 
        return
    end

    % K*alpha = y;
    % L'*L*alpha = y;
    alpha = L'\(L\y);

    % Log Likelihood
    LOGLIK = (-0.5*y'*alpha - sum(log(diag(L))) - log(2*pi)*sum(n)/2);
end