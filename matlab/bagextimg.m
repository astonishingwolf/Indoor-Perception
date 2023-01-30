files = dir(['C:\Users\dasgu\Downloads\Sept_Recording\Sep28-2022_CAM-LCR_LIDAR_applanix\14-50-47\*.bag']); % Enter the bag file location here
n = length(files);
disp(n);
for k = 1:n
    %print('het');
    disp(files(k).name)
    bagselect = rosbag(files(k).name);
    bSel = select(bagselect,'Topic','/pylon_camera_node_center/image_rect/compressed');
    msgStructs = readMessages(bSel,'DataFormat','struct');
    pt = cell(length(msgStructs),1);
    for i = 1:length(msgStructs)
        img = rosReadImage(msgStructs{i,1});
        img = imresize(img,1/3);
        pt{i} = img;
    end
    save(append("imgdata/2022-06-16-13-01-40_",string(k),"_img.mat"),'pt','-v7.3');
    fprintf('%.2f done\n',k/n);
end