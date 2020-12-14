function niqe_value = niqe(root_dir)
    path_list = dir(strcat(root_dir, '*.png'));

    if isempty(path_list) == 0
        path_list = dir(strcat(root_dir, '*.png'));
    end
    
    img_num = length(path_list);
    niqe_value = 0;
    if img_num > 0 
        for k = 1:img_num
            image_name = convertCharsToStrings(path_list(k).name);
            input = imread(strcat(root_dir, image_name));
            niqe_value = niqe_value + niqe(input);
        end
        niqe_value = niqe_value / img_num;
    end

    writematrix(niqe_value, './niqe_data.txt')
    exit
end
