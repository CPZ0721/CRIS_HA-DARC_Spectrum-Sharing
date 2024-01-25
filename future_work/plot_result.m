clear all; close all;

user_2 = [1274808.684,	2724599.842,	3992619.888,	5755492.777,	6720503.456,	8556526.649];
user_3 = [1146737.735,	2501754.492,	3718280.091,	5218605.519,	6566007.716,	7653157.396];
user_4 = [1032691.921,	2252245.719,	3480157.648,	4898350.116,	6399126.703,	7415844.848];

x = [1,2,3,4,5,6];


% 製作圖表
figure;
plot(x, user_2, 'Color',[.929, .694, .125], 'LineStyle', '-','Marker','v');
hold on;
plot(x, user_3,'Color',[0 0.4470 0.7410], 'LineStyle', '-','Marker','^');
hold on;
plot(x, user_4, 'Color',[0.4660 0.6740 0.1880], 'LineStyle', '-','Marker','o');

xticks(x);
% xticklabels({'1W', '2W', '3W', '4W', '5W', '6W'});

ylabel('The Capcity of All Secondary Users (bps)');
legend("The number of SU=2",'The number of SU=3',"The number of SU=4","location","southeast");
xlabel('The number of idle spectrum');
