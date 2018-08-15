%% This code follows the Sigmund 2018 infill bone paper
% clear;close all;
% addpath('./fminsdp/');
load('experiment_data/LHS_train.mat')
load('experiment_result/phi_gen.mat')
load('experiment_result/random_candidate.mat')
random_candidate=random_candidate+1; % python to matlab
%% Input
ratio=10;
nelx=12*ratio; % horizontal length
nely=4*ratio; % vertical length
alpha =0.6; % lobal volume fraction
alpha2=0.6; % global volume fraction
gamma=3.0; % material binarization
rmin=3.0; % filter radius
density_r = 6.0; % density radius

batch_size=100;
LHS_rand=LHS_train(random_candidate,:);

xPhys_true = zeros(100,nely,nelx);
% phi_true = zeros(batch_size,nely,nelx);
% c_store= zeros(batch_size,1);
c_our_final=zeros(batch_size,1);
mu_store=zeros(batch_size,1);
rho_store=zeros(batch_size,4800);

for iii = 1:1:batch_size
    
count=0;
force=-1;

% random force location in right range
F = sparse(2*(nely+1)*(nelx+1),1);

%% LHS
point_rand = ((nely+1)*(LHS_rand(iii,1)-1)+LHS_rand(iii,2))*2;
theta_rand=LHS_rand(iii,3);
Fx=force*sin(theta_rand);
Fy=force*cos(theta_rand);
F(point_rand-1,1)= Fx;
F(point_rand,1)= Fy;

%% Algorithm parameters
p = 16; 
beta=8.0; 
nn = nelx*nely;
epsilon_al=1e-3;
epsilon_opt=1e-3;

% PREPARE FILTER
% create neighbourhood index for M (filtering)
r = rmin;
range = -r:r;
[X,Y] = meshgrid(range,range);
neighbor = [X(:), Y(:)];
D = sum(neighbor.^2,2);
rn = sum(D<=r^2 & D~=0);
pattern = neighbor(D<=r^2 & D~=0,:);
[locX, locY] = meshgrid(1:nelx,1:nely);
loc = [locY(:),locX(:)];
M = zeros(nn,rn);
for i = 1:nn
    for j = 1:rn
        locc = loc(i,:) + pattern(j,:);
        if sum(locc<1)+sum(locc(1)>nely)+sum(locc(2)>nelx)==0
            M(i,j) = locc(1) + (locc(2)-1)*nely;
        else
            M(i,j) = NaN;
        end
    end
end
% M_t = M';
% idx = kron((1:nn)',ones(rn,1));
% idy = M_t(~isnan(M_t));
% idx(isnan(M_t))=[];
% bigM = sparse(idx,idy,1);

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
bigM = H>0;

% create neighbourhood index for N (local density)
r = density_r;
range = -r:r;
mesh = range;
[X,Y] = meshgrid(range,range);
neighbor = [X(:), Y(:)];
D = sum(neighbor.^2,2);
rn = sum(D<=r^2);
pattern = neighbor(D<=r^2,:);
[locX, locY] = meshgrid(1:nelx,1:nely);
loc = [locY(:), locX(:)];
N = zeros(nn,rn);
for i = 1:nn
    for j = 1:rn
        locc = loc(i,:) + pattern(j,:);
        if sum(locc<1)+sum(locc(1)>nely)+sum(locc(2)>nelx)==0
            N(i,j) = locc(1) + (locc(2)-1)*nely;
        else
            N(i,j) = NaN;
        end
    end
end
idx = kron((1:nn)',ones(rn,1));
N_t = N';
idy = N_t(~isnan(N_t));
idx(isnan(N_t))=[];
bigN = sparse(idx,idy,1);
N_count = sum(~isnan(N),2);

%% MATERIAL PROPERTIES
E0 = 1;
Emin = 1e-9;
nu = 0.3;
% xPhys_true(iii,:,:)=reshape(rho,[nely,nelx]);
% phi_true(iii,:,:)=reshape(phi,[nely,nelx]);
% c_store(iii,:)=c;
% figure,colormap(gray); imagesc(1-reshape(rho,[nely,nelx])); caxis([0 1]); axis equal; axis off; drawnow;
% fprintf(' It.:%5i Obj.:%11.4f g:%7.3f eta:%7.3f r:%7.3f ch.:%7.3f\n',count, c, g, eta, log(r), max(abs(full(dphi))));
% saveas(gcf,sprintf('FIG_show_%d.png',iii));
% close(1)
% count_store(iii,:)=count;

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
% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)

% F = sparse((nely+1)*2*nelx+nely+2,1,2,2*(nely+1)*(nelx+1),1);

U = zeros(2*(nely+1)*(nelx+1),1);
fixeddofs = 1:2*(nely+1);
alldofs = [1:2*(nely+1)*(nelx+1)];
freedofs = setdiff(alldofs,fixeddofs);

%prepare some stuff to reduce cost
dphi_idphi = bsxfun(@rdivide,H,sum(H));

%% START ITERATION
loop = 0;

phi = reshape(double(phi_gen(iii,:)),[nn,1]);

r = 1;
r2 = 0.001;

loop=loop+1;
% for iii = 1:1   
%     loop = loop + 1;
loop2 = 0;
%augmented lagrangian parameters
r = r*2;
r2 = r2*2;
lambda = 0;
lambda2 = 0;
eta = 0.1;
eta2 = 0.1;
epsilon = 1e-3;
learning_rate = 0.1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Get initial g and c %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phi_til = H*phi(:)./Hs;  
rho = (tanh(beta/2)+tanh(beta*(phi_til-0.5)))/(2*tanh(beta/2));
%     rho=double(reshape(rho_test(iii,:),[nn,1]));

sK = reshape(KE(:)*(Emin+rho(:)'.^gamma*(E0-Emin)),64*nelx*nely,1);
K = sparse(iK,jK,sK); K = (K+K')/2; 
U(freedofs) = K(freedofs,freedofs)\F(freedofs);
ce = sum((U(edofMat)*KE).*U(edofMat),2);
c = sum(sum((Emin+rho(:).^gamma*(E0-Emin)).*ce));  
c_our_final(iii,:)=c;
rho_bar = bigN*rho./N_count;
g = (sum(rho_bar.^p)/nely/nelx)^(1/p)/alpha - 1.0;
global_density = rho'*ones(nn,1)/nn-alpha2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dc_drho = -gamma*rho.^(gamma-1)*(E0-Emin).*ce(:); 
drho_dphi = beta*(1-tanh(beta*(phi(:)-0.5)).^2)/2/tanh(beta/2);
dc_dphi = sum(bigM.*bsxfun(@times, dphi_idphi, (dc_drho.*drho_dphi)'),2);
dg_drhobar = 1/alpha/nn*(1/nn*sum(rho_bar.^p)).^(1/p-1).*rho_bar.^(p-1);
dg_dphi = sum(bigM.*bsxfun(@times, dphi_idphi, (bigN*(dg_drhobar./N_count).*drho_dphi)'),2);

    
dg_dphi=dg_dphi';
dc_dphi=dc_dphi';

mu_check=full(sum((dc_dphi*(1-dg_dphi'*(dg_dphi*dg_dphi'+eye(1)*1e-12)^(-1)*dg_dphi)).^2));
mu_store(iii,:)=mu_check;   
rho_store(iii,:)=rho(:);
fprintf('evaluating sample %s',i)
end

[B,I]=sort(mu_store,'descend');
add_point_index=random_candidate(I(1)-1); % matlab to python
save(sprintf('experiment_result/add_point_index.mat'),'add_point_index');
