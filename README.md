# halo_vpf
Void probablitity function for cosmological applications

BolshoiTraditionalTwoPoint.ipynb contains functions to calculate 2-point correlation function, matter and halo power spectrum, void probability function (VPF-1NN).
PaperNotebook.ipynb contains functions to calculate kNN VPF.

Scripts for the paper

1. SZ_r_sher.py: generates distance based kNN measurements for 1x Planck SZ catalog and 2000 (adjustable using variable 'len_poi') random Poisson realizations. Designed for usage on sherlock. Default k set to [1, 5, 10, 50, 2, 3, 4, 8, 5, 9]. Default file output directory '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ'.

2. get_CDF_1248.py: Reads in the various kNN distances. Interpolates the CDF for Planck SZ and Poisson samples. Stores the interpolated function values in the form of peaked CDF for the Planck SZ sample. Store the mean and standard deviation in the form of peaked CDFs for the Poisson samples. Default file output directory: '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/CDF'.

3. get_CIC_1248.py: Reads in the various kNN distances. Interpolates the CDF for Planck SZ and Poisson samples. Subtracts relative CDFs to derive CICs. Default file output directory: '/oak/stanford/orgs/kipac/users/ycwang19/VPF/SZ/CIC'.

4. plot_CDFs.py: Reads in data saved in step 2. Plots the left column of Figure 3. Saves figure as 'CDFs_test.png'.

5. plot_CICs.py: Reads in data saved in step 3. Plots the right column of Figure 3. Saves figure as 'CICs_test.png'.

6. plot_CDF_CIC.py: Reads in the pngs generated in steps 4 and 5. Plots the combined Figure 3 as 'CDF_CIC.pdf'.

7. join_3D_mask.py: Plots the galactic latitude and longitude of the Planck SZ sample and one Poisson realization. Plots Figure 1 as 'join_3D_mask_Y500.pdf'.

8. Msz_z_Y500.py: Plots the redshift and SZ mass distributions in Figure 2 as 'SZ_dist.pdf'.

9. xi_Y500_1.py: Generates the two-point correlation function measurements for Figure 4. xi(r) generated using corrfunc 'convert_3d_counts_to_cf', number of Poisson realizations set by the variable 'len_poi', default 1000 samples. RR randoms assumed to be Poisson randoms also following the Planck SZ sample's redshift distribution, but the 't' times the mean number density, where t is set to 10 by default. Outputs Figure 4 as 'xi_Y500_{len_poi}.pdf'.

10. cov_xi_test.py: Calculates the same set of xi(r) as in step 9. Calculates chi^2 for Planck SZ and chi^{2} distribution of the random Poisson samples. Covariance matrix saved as 'xi_cov_{len_poi}.txt'. Produces lower panel of Figure 5.

11. cdf_cov_test.py: Calculates the chi_{2} and covariance matrices from the data saved in step 2. Covariance matrix saved as 'cdf_cov_{len_poi}.txt'. Produces upper panel of Figure 5.

12. plot_chi2.py: Combines the two figure outputs in step 10 and 11. Saves figure as 'chi2.pdf', which is Figure 5 in the draft.

13. plot_cov_xi.py: Reads in covariance matrix saved in step 10 for xi(r). Calculates the correlation matrix, plots the right panel in Figure 6.

14. plot_cdf_xi.py: Reads in covariance matrix saved in step 11 for kNN CDFs. Calculates the correlation matrix, plots the left panel in Figure 6.

15. plot_cov_comb.py: Combines the figure outputs in step 13 and 14. Plots Figure 6 in the draft as 'cov.pdf'.
