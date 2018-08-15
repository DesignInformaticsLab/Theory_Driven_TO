%% This code follows the Sigmund 2018 infill bone paper
function infill_high_dim(initial)
fprintf('start solving the TO problem, intial is %d......\n',initial)

%% Input
ratio=10;
nelx=12*ratio; % horizontal length
nely=4*ratio; % vertical length
alpha =0.6; % lobal volume fraction
alpha2=0.6; % global volume fraction
gamma=3.0; % material binarization
rmin=3.0; % filter radius
density_r = 6.0; % density radius


LHS=load('experiment_data/LHS_train.mat');
LHS=LHS.LHS_train;
if initial~=1
    index=load('experiment_result/random_candidate.mat')+1; % python to matlab
    index=index.random_cadidate+1;
    phi_true_train=load('experiment_data/phi_true_train.mat');
    phi_true_train=phi_true_train.phi_true_train;
else
    index=load('experiment_result/index_ind.mat')+1; % python to matlab
    index=index.index_ind+1;
    phi_true_train=zeros(length(LHS),nelx*nely);
end
LHS=LHS(index);

batch_size=length(index);
xPhys_true = zeros(batch_size,nely,nelx);
phi_true = zeros(batch_size,nely,nelx);
c_store= zeros(batch_size,1);

budget_store=zeros(batch_size,1);
% force_store=zeros(nelx,batch_size);
% theta_store=zeros(1,batch_size);
% point_store=zeros(1,batch_size);


for iii = 1:1:batch_size
    
budget=0;
force=-1;

%% LHS
F = sparse(2*(nely+1)*(nelx+1),1);
point_rand = ((nely+1)*(LHS(iii,1)-1)+LHS(iii,2))*2;
theta_rand=LHS(iii,3);
Fx=force*sin(theta_rand);
Fy=force*cos(theta_rand);
F(point_rand-1,1)= Fx;
F(point_rand,1)= Fy;

%% Algorithm parameters
p = 16; 
beta=1.0; 
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
phi = alpha*ones(nn,1);
% phi = double(reshape(phi_test(1,:),[nn,1]));
loop = 0;
delta = 1.0;

r = 1;
r2 = 0.001;
while beta < 10
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
    delta = 1e6;
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
    rho_bar = bigN*rho./N_count;
    g = (sum(rho_bar.^p)/nely/nelx)^(1/p)/alpha - 1.0;
    global_density = rho'*ones(nn,1)/nn-alpha2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    phi_old = phi;
    
    while 1
        if (max(abs(full(delta)))<epsilon_al && g<0.05 && loop2>0) 
            break;
        end
        loop2 = loop2 + 1;
        delta = 1e6;
        j=0;j2=0;
        loop3 = 0;
       
        while max(abs(full(delta))) > epsilon && loop3 < 200
%             max(abs(full(dphi)))
            loop3 = loop3 + 1;
            learning_rate = 0.1;

            dc_drho = -gamma*rho.^(gamma-1)*(E0-Emin).*ce(:); 
            drho_dphi = beta*(1-tanh(beta*(phi(:)-0.5)).^2)/2/tanh(beta/2);
            dc_dphi = sum(bigM.*bsxfun(@times, dphi_idphi, (dc_drho.*drho_dphi)'),2);
            dg_drhobar = 1/alpha/nn*(1/nn*sum(rho_bar.^p)).^(1/p-1).*rho_bar.^(p-1);
            dg_dphi = sum(bigM.*bsxfun(@times, dphi_idphi, (bigN*(dg_drhobar./N_count).*drho_dphi)'),2);
            dphi = dc_dphi + lambda*dg_dphi*(g>0) + 2/r*g*dg_dphi*(g>0) +...
                lambda2*drho_dphi/nn*(global_density>0) + 2/r2*global_density*drho_dphi/nn*(global_density>0);
            
%             mu_check=sum((dc_dphi*(1-dg_dphi'*(dg_dphi*dg_dphi')*dg_dphi)).^2);
%             fprintf('mu_check.:%11.4f beta:%7.3f\n',sum(full(mu_check)), beta);
            
            % check if learning rate is too large
            g_old = g;
            global_density_old = global_density;
            c_old = c;
            loop4 = 0;
            while loop4 < 100
                budget=budget+1;
                delta = max(-0.1*ones(nn,1),min(0.1*ones(nn,1),-dphi*learning_rate));
                phi_temp = max(zeros(nn,1),min(ones(nn,1),phi + delta));
%                 delta = -dphi*learning_rate;
%                 phi_temp = max(zeros(nn,1),min(ones(nn,1),phi + delta));
%                 phi_temp = phi + delta;
%                 delta = phi_temp - phi;
                
                phi_til = H*phi_temp(:)./Hs;  
                rho_temp = (tanh(beta/2)+tanh(beta*(phi_til-0.5)))/(2*tanh(beta/2));
                rho_bar_temp = bigN*rho_temp./N_count;
                g_temp = (sum(rho_bar_temp.^p)/nely/nelx)^(1/p)/alpha - 1.0;
                global_density_temp = rho_temp'*ones(nn,1)/nn-alpha2;
                sK = reshape(KE(:)*(Emin+rho_temp(:)'.^gamma*(E0-Emin)),64*nelx*nely,1);
                K = sparse(iK,jK,sK); K = (K+K')/2; 
                U(freedofs) = K(freedofs,freedofs)\F(freedofs);
                ce_temp = sum((U(edofMat)*KE).*U(edofMat),2);
                c_temp = sum(sum((Emin+rho_temp(:).^gamma*(E0-Emin)).*ce_temp));
                dobj = (c_temp + lambda*g_temp*(g_temp>0) + 1/r*g_temp^2*(g_temp>0) + lambda2*global_density_temp*(global_density_temp>0) + 1/r2*global_density_temp^2*(global_density_temp>0))-...
                    (c_old + lambda*g_old*(g_old>0) + 1/r*g_old^2*(g_old>0) + lambda2*global_density_old*(global_density_old>0) + 1/r2*global_density_old^2*(global_density_old>0));
                if dobj>0
                    learning_rate = learning_rate*0.5;
                    loop4 = loop4 + 1;
                    if loop4 == 100
                        loop3 = 10000;
                        delta = 0;
                    end
                else
                    phi = phi_temp;
                    c = c_temp;
                    g = g_temp;
                    ce = ce_temp;
                    rho = rho_temp;
                    rho_bar = rho_bar_temp;
                    global_density = global_density_temp;
                    break;
                end
            end
            %% PRINT RESULTS
%             fprintf(' It.:%5i Obj.:%11.4f g:%7.3f eta:%7.3f r:%7.3f ch.:%7.3f beta:%7.3f\n',loop, c, ...
%             g, eta, log(r), max(abs(full(delta))),beta);
            %% PLOT DENSITIES
%             colormap(gray); imagesc(1-reshape(rho,[nely,nelx])); caxis([0 1]); axis equal; axis off; drawnow;
        end
        
        if g<eta
            lambda = lambda + 2*g/r;
            j = j+1;
            %         epsilon = r^(j+1);
            eta = eta*0.5;
        else
            r = 0.5*r;
            j = 0;
        end
        
        if global_density<eta2
            lambda2 = lambda2 + 2*global_density/r2;
            j2 = j2+1;
            %         epsilon = r^(j+1);
            eta2 = eta2*0.5;
        else
            r2 = 0.5*r2;
            j2 = 0;
        end
    end    
    
    delta = full(max(abs(phi-phi_old)));
    % step 17
%     if mod(loop,40) == 0 || delta < epsilon_opt
    beta = 2*beta;
%     delta = 1.0;
%     end
end

xPhys_true(iii,:,:)=reshape(rho,[nely,nelx]);
phi_true(iii,:,:)=reshape(phi,[nely,nelx]);
c_store(iii,:)=c;
budget_store(iii,:)=budget;
figure(1),colormap(gray); imagesc(1-reshape(rho,[nely,nelx])); caxis([0 1]); axis equal; axis off; drawnow;
% colormap(gray); imagesc(1-rho); caxis([0 1]); axis equal; axis off; drawnow;
% fprintf(' It.:%5i Obj.:%11.4f g:%7.3f eta:%7.3f r:%7.3f ch.:%7.3f iii:%7.3f \n',budget, c, g, eta, log(r), max(abs(full(dphi))),iii);
% saveas(gcf,sprintf('%s/FIG_show_%d.png',fname1,iii));
% close(1)
% iii
end
fname1='experiment_data';
% fname2='experiment_result';
for index_LHS = 1:batch_size
    phi_true_train(index(index_LHS),:)=phi_true(index_LHS).reshape(nely*nexl,1);
end

save(sprintf('%s/budget_store.mat',fname1),'budget_store');
save(sprintf('%s/phi_true_train.mat',fname1),'phi_true_train');
% save(sprintf('%s/xPhys_true.mat',fname1),'xPhys_true');
% save(sprintf('%s/c_store.mat',fname1),'c_store');
