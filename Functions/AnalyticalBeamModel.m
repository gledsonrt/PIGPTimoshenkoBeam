function AP = AnalyticalBeamModel(AP)
    % Simply supported beam with uniform distributed load
    
    % Response
    AP.v = AP.q*AP.x - AP.q*AP.L/2;
    AP.m = AP.q*AP.x.^2/2 - AP.q*AP.L*AP.x/2;
    AP.s = -AP.z*(AP.m/AP.EI);
    AP.p = (AP.q*AP.x.^3/6 - AP.q*AP.L*AP.x.^2/4 + AP.q*AP.L^3/24)/AP.EI;
    AP.a = (AP.q*AP.x.^4/24 - AP.q*AP.L*AP.x.^3/12 + AP.q*AP.L^3*AP.x/24)/AP.EI;
    AP.e = AP.s + AP.z*AP.q/AP.kGA;
    AP.r = AP.p - AP.v/AP.kGA;
    AP.w = AP.a - AP.m/AP.kGA;  
    AP.f = AP.q*ones(size(AP.x));
end