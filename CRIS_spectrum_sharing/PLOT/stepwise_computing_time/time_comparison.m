clear all; close all;

time_compare = csvread("time_compare.csv")
DARC_time = time_compare(:,1);
bb_time = time_compare(:,2);
sa_time = time_compare(:,3);

x = 0:1:40;
plot(x, bb_time, 'Color',[0 0.4470 0.7410], 'LineStyle', '-','Marker','^');
hold on;
plot(x, sa_time,'Color',[0.4940 0.1840 0.5560], 'LineStyle', '-','Marker','*');
hold on;
plot(x, DARC_time,'Color',[0.8500 0.3250 0.0980], 'LineStyle', '-','Marker','o');
hold on;
ylabel('Time (s)');
xlabel('Time Steps');
set(gca, 'YScale', 'log');
legend('Branch and bound','Heuristic-SA','HA-DARC',"location", "best");