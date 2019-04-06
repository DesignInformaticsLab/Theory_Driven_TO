%% This code follows the Sigmund 2018 infill bone paper

function [c, g, global_density] = calculate_compliance(x, y)
    % x is the angle
    % y is the topology

    %% Input
    ratio=10;
    nelx=12*ratio; % horizontal length
    nely=4*ratio; % vertical length
    alpha =0.6; % lobal volume fraction
    alpha2=0.6; % global volume fraction
    gamma=3.0; % material binarization
    rmin=3.0; % filter radius
    density_r = 6.0; % density radius

    force=-1;
    F = sparse(2*(nely+1)*(nelx+1),1);

    Fx=force*sin(x);
    Fy=force*cos(x);
    F(2*(nely+1)*nelx+nely+1,1)= Fx;
    F(2*(nely+1)*nelx+nely+2,1)= Fy;

    %% Algorithm parameters
    p = 16; % local density constraint
    beta=16; % density transformation
    nn = nelx*nely; % topology dimension

    %% FILTER
    iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
    jH = ones(size(iH));
    sH = zeros(size(iH));
    k = 0;
    for i1 = 1:nelx
      for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
          for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
            e2 = (i2-1)*nely+j2;
            k = k+1;
            iH(k) = e1;
            jH(k) = e2;
            sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
          end
        end
      end
    end
    H = sparse(iH,jH,sH);
    Hs = sum(H,2);

    %% MATERIAL PROPERTIES
    E0 = 1;
    Emin = 1e-9;
    nu = 0.3;

    %% PREPARE FINITE ELEMENT ANALYSIS
    A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
    A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
    B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
    B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
    KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
    nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
    edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
    edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
    iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
    jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
    U = zeros(2*(nely+1)*(nelx+1),1);
    fixeddofs = 1:2*(nely+1);
    alldofs = [1:2*(nely+1)*(nelx+1)];
    freedofs = setdiff(alldofs,fixeddofs);

    %% PREPROCESS INPUT
    phi_til = H*y(:)./Hs;
    rho = (tanh(beta/2)+tanh(beta*(phi_til-0.5)))/(2*tanh(beta/2));

    %% FINITE ELEMENT ANALYSIS
    sK = reshape(KE(:)*(Emin+rho(:)'.^gamma*(E0-Emin)),64*nelx*nely,1);
    K = sparse(iK,jK,sK); K = (K+K')/2;
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);
    ce = sum((U(edofMat)*KE).*U(edofMat),2);
    c = sum(sum((Emin+rho(:).^gamma*(E0-Emin)).*ce));

    %% CONSTRAINT VIOLATION
    g = (sum(rho_bar.^p)/nely/nelx)^(1/p) - alpha;
    global_density = rho'*ones(nn,1)/nn-alpha2;
