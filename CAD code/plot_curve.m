x = linspace(1,100,100)';

figure,
line(x,mu_c_data,'Color','g','LineWidth',1.5,'LineStyle','-');hold on;
line(x,mu_c_data2,'Color','b','LineWidth',1.5,'LineStyle','-');hold on
line(x,mu_c_test,'Color','k','LineWidth',1.5,'LineStyle','-','Marker','s');
line(x,mu_c,'Color','r','LineWidth',2.0);
legend('benchmark I','benchmark II','proposed method','ground truth')

fill([x;flipud(x)],[mu_c_data-var_c_data;flipud(mu_c_data+var_c_data)],'g','linestyle','none');
alpha(0.25)
fill([x;flipud(x)],[mu_c_data2-var_c_data2;flipud(mu_c_data2+var_c_data2)],'b','linestyle','none');
alpha(0.25)
fill([x;flipud(x)],[mu_c_test-var_c_test;flipud(mu_c_test+var_c_test)],'k','linestyle','none');
alpha(0.25)
set(gca, 'xtick', 0:25:100)
set(gca,'fontsize',14)
box on


%% error plot

x = linspace(1,100,100)';

c_true=mu_c;
error=abs(c_true-mu_c_test);
error_data=abs(c_true-mu_c_data);
error_data2=abs(c_true-mu_c_data2);

figure,
line(x,error_data,'Color','g','LineWidth',1.5,'LineStyle','-');hold on;
line(x,error_data2,'Color','b','LineWidth',1.5,'LineStyle','-');hold on
line(x,error,'Color','k','LineWidth',1.5,'LineStyle','-','Marker','s');
% line(x,c_true,'Color','r','LineWidth',2.0);
legend('benchmark I - Mean:26.5; Std:27.1','benchmark II - Mean:10.4; Std:9.0','proposed method - Mean:6.2; Std:5.5')

fill([x;flipud(x)],[error_data-var_c_data;flipud(error_data+var_c_data)],'g','linestyle','none');
alpha(0.25)
fill([x;flipud(x)],[error_data2-var_c_data2;flipud(error_data2+var_c_data2)],'b','linestyle','none');
alpha(0.25)
fill([x;flipud(x)],[error-var_c_test;flipud(error+var_c_test)],'k','linestyle','none');
alpha(0.25)
ylim([-100 300])
set(gca, 'xtick', 0:25:100)
set(gca, 'ytick', -100:50:300)
set(gca,'fontsize',14)
box on

% %% error mean
% error_sum=mean(error);
% error_var=std(error);
% error_data_sum=mean(error_data);
% error_data_var=std(error_data);
% error_data2_sum=mean(error_data2);
% error_data2_var=std(error_data2);
% 
% x = 1:1:3;
% y = [error_data_sum,error_data2_sum,error_sum];
% err = [error_data_var,error_data2_var,error_var];
% figure,
% h=errorbar(x,y,err,'s','MarkerEdgeColor','red','MarkerFaceColor','red','linewidth',4);
% h.CapSize = 12;