clear all, close all, clc

average_HSV_IR = load('HSV+IR_results.mat', 'average_result');
average_HSV_IR = average_HSV_IR.average_result;
average_HSV_IR_2 = load('HSV+IR_2_results.mat', 'average_result');
average_HSV_IR_2 = average_HSV_IR_2.average_result;
average_HSV_IR_normalized = load('HSV+IR_normalized_results.mat', 'average_result');
average_HSV_IR_normalized = average_HSV_IR_normalized.average_result;
average_Y_IR = load('Y+IR_results.mat', 'average_result');
average_Y_IR = average_Y_IR.average_result;

figure, 
hold on
plot(average_HSV_IR(4:6))
plot(average_HSV_IR_2(4:6))
plot(average_HSV_IR_normalized(4:6))
plot(average_Y_IR(4:6))
legend('HSV+IR', 'HSV+IR2', 'HSV+IRnorm', 'Y+IR')