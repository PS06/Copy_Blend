function niqe_value = niqe_file(image_file)
    image = convertCharsToStrings(image_file);
    input = imread(image);
    niqe_value = niqe(input);
    writematrix(niqe_value, './niqe_data.txt')
    exit
end