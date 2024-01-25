clear all; close all;

bb = [699.882413134996,	729.9013516635932,	768.8336309,	797.6914155];
sa = [678.5097283,	700.4441182,	719.6281594,	728.4329926];

DARC = [635.362248334596, 654.3061419282601,669.484017962046,690.93932159585];
DDPG=[605.9174696147836,643.818258752221,654.072757353828,679.2816715014444];

random = [539.521081,	564.750044,	580.2935381, 609.1881007];
x = [8,12,16,20];


% 製作圖表
figure;
plot(x, random, 'Color',[.929, .694, .125], 'LineStyle', '-','Marker','v');
hold on;
plot(x, bb,'Color',[0 0.4470 0.7410], 'LineStyle', '-','Marker','^');
hold on;
plot(x, sa, 'Color',[0.4660 0.6740 0.1880], 'LineStyle', '-','Marker','o');
hold on;
plot(x, DDPG,'Color',[0.4940 0.1840 0.5560], 'LineStyle', '-','Marker','<');
hold on;
plot(x, DARC,'Color',[0.8500 0.3250 0.0980], 'LineStyle', '-','Marker','*');

hold on;
ylabel('The Spectral Efficiency of All Secondary Users (bps/Hz)');
legend("Best Random",'Branch and bound',"Heuristic-SA","HA-DDPG","HA-DARC","location", "southeast");
xlabel('The number of RIS reflective elements');
