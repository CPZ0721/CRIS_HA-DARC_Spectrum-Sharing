clear all; close all;

result_compare = csvread("result_compare.csv")
DARC_result= result_compare(:,1);
bb_result = result_compare(:,2);
sa_result = result_compare(:,3);

x = 0:1:40;
plot(x, bb_result, 'Color',[0 0.4470 0.7410], 'LineStyle', '-','Marker','^');
hold on;
plot(x, sa_result,'Color',[0.4940 0.1840 0.5560], 'LineStyle', '-','Marker','*');
hold on;
plot(x, DARC_result,'Color',[0.8500 0.3250 0.0980], 'LineStyle', '-','Marker','o');
hold on;
ylabel('The Spectral Efficiency of All Secondary Users (bps/Hz)');

xlabel('Time Steps');
ylim([0 43])
legend('Branch and bound','Heuristic-SA','HA-DARC',"location", "best");