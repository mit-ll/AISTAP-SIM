% Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
% Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
% SPDX-License-Identifier: MIT

pathname = '';
filename = [pathname, 'simMed/simMed_sample.mat']; % rename as desired

r = load(filename);

disp('Display one channel')
img_ind = 1;
chan_ind = 1;
figure(1);clf;
imagesc(db(r.rd_img(:,:,chan_ind,img_ind))); caxis([-10,30]);
xlabel('Doppler'); ylabel('range');

hold on;
plot(r.meta_per_image{1}.targ_pix_dop + metadata.midp_dop, r.meta_per_image{1}.targ_pix_range + metadata.midp_range,'r.')
hold on;


disp('Display the sum channel')
img_ind = 1;
chan_ind = 1;
figure(2);clf;
imagesc(db(sum(r.rd_img(:,:,:,img_ind),3))); caxis([-10,30]);
xlabel('Doppler'); ylabel('range');

hold on;
plot(r.meta_per_image{1}.targ_pix_dop + metadata.midp_dop, r.meta_per_image{1}.targ_pix_range + metadata.midp_range,'r.')
hold on;

