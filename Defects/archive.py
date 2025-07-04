








def make_big_plot(plot_defect_type:str, side_length:int, clipHighestLowest:bool = False):


    # -------------------------------------
    if plot_defect_type == "substitution":
        Lattice = DefectSquareLattice(side_length, plot_defect_type)

        M_back_vals = [-2.5, -1.0, 1.0, 2.5]
        M_sub_vals = [-2.5, -1.0, 1.0, 2.5]
        n_rows, n_cols = 3, 2 * len(M_back_vals)
        scale = 4
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(scale*n_cols, scale*n_rows))  

        for i, M_background in enumerate(M_back_vals):
            spectrum_axs = axs[:, 0+2*i].flatten()
            ldos_axs = axs[:, 1+2*i].flatten()
            M_sub_vals_removed = M_sub_vals.copy()
            M_sub_vals_removed.remove(M_background)
            for spectrum_ax, ldos_ax, M_substitution in zip(spectrum_axs, ldos_axs, M_sub_vals_removed):
                hamiltonian = Lattice.compute_hamiltonian(M_background, M_substitution)
                LDOS, eigenvalues, gap = Lattice.compute_LDOS(hamiltonian, number_of_states=2, returnEigenvalues=True, returnGap=True)
                ldos_ax.scatter(Lattice.X, Lattice.Y, c=LDOS, s=50, cmap='inferno')
                ldos_ax = format_ldos_ax(ldos_ax, rf"$m_0^{{sub}}$ = {M_substitution}")

                spectrum_ax.scatter(np.arange(len(eigenvalues)), eigenvalues, label="Gap = {:.2e}".format(gap))
                spectrum_ax.set_xlabel("Eigenvalue Index")
                spectrum_ax.set_ylabel("Energy")
                spectrum_ax.set_title(f"Spectrum")
                spectrum_ax.legend()
                #spectrum_ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

                if ldos_ax.collections:  # Check if collections exist
                    norm = plt.Normalize(vmin=max(1e-10, min(coll.get_array().min() for coll in ldos_ax.collections)),
                                            vmax=max(coll.get_array().max() for coll in ldos_ax.collections))
                    for coll in ldos_ax.collections:
                        coll.set_norm(norm)
                    
                cbar = fig.colorbar(ldos_ax.collections[0], ax=ldos_ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label("LDOS")
                cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))

                fig.text((2*i+1)/8, 0.95, rf"$m_0^{{back}}$ = {M_background}", fontsize=20, ha='center')
          
        formatter = ticker.FormatStrFormatter('%.1e')
        for ax in axs.flatten():
            for im in ax.collections:
                if hasattr(im, 'colorbar') and im.colorbar is not None:
                    im.colorbar.ax.yaxis.set_major_formatter(formatter)


        fig.suptitle(rf"Substitution", fontsize=30)
        direc="./Defects/Plots/LDOS/"
        plt.tight_layout()
        plt.savefig(direc+f"LDOS_{Lattice.defect_type}_{side_length}_all.png")

    # -------------------------------------
    elif plot_defect_type == "interstitial":
        Lattice = DefectSquareLattice(side_length, plot_defect_type)

        M_back_vals = [-2.5, -1.0, 1.0, 2.5]
        M_sub_vals = [-2.5, -1.0, 1.0, 2.5]
        n_rows, n_cols = len(M_sub_vals), 2 * len(M_back_vals)
        scale = 4
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(scale*n_cols, scale*n_rows))   

        for i, M_background in enumerate(M_back_vals):  
            spectrum_axs = axs[:, 0+2*i].flatten()
            ldos_axs = axs[:, 1+2*i].flatten()
            for spectrum_ax, ldos_ax, M_substitution in zip(spectrum_axs, ldos_axs, M_sub_vals):
                hamiltonian = Lattice.compute_hamiltonian(M_background, M_substitution)
                LDOS, eigenvalues, gap = Lattice.compute_LDOS(hamiltonian, number_of_states=2, returnEigenvalues=True, returnGap=True)
                if clipHighestLowest:
                    LDOS = np.clip(LDOS, np.sort(LDOS)[1], np.sort(LDOS)[-2])
                ldos_ax.scatter(Lattice.X, Lattice.Y, c=LDOS, s=50, cmap='inferno')
                ldos_ax = format_ldos_ax(ldos_ax, rf"$m_0^{{sub}}$ = {M_substitution}")
    
                spectrum_ax.scatter(np.arange(len(eigenvalues)), eigenvalues, label="Gap = {:.2e}".format(gap))
                spectrum_ax.set_xlabel("Eigenvalue Index")
                spectrum_ax.set_ylabel("Energy")
                spectrum_ax.set_title(f"Spectrum")
                spectrum_ax.legend()
                #spectrum_ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

                if ldos_ax.collections:  # Check if collections exist
                    norm = plt.Normalize(vmin=max(1e-10, min(coll.get_array().min() for coll in ldos_ax.collections)),
                                            vmax=max(coll.get_array().max() for coll in ldos_ax.collections))
                    for coll in ldos_ax.collections:
                        coll.set_norm(norm)
                    
                cbar = fig.colorbar(ldos_ax.collections[0], ax=ldos_ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label("LDOS")
                cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))
            fig.text((2*i+1)/8, 0.95, rf"$m_0^{{back}}$ = {M_background}", fontsize=20, ha='center')


        formatter = ticker.FormatStrFormatter('%.1e')
        for ax in axs.flatten():
            for im in ax.collections:
                if hasattr(im, 'colorbar') and im.colorbar is not None:
                    im.colorbar.ax.yaxis.set_major_formatter(formatter)

        if not clipHighestLowest:
            fig.suptitle("Interstitial", fontsize=30)
        else:
            fig.suptitle("Interstitial (clipped)", fontsize=30)

        plt.tight_layout()

        direc="./Defects/Plots/LDOS/"
        if not clipHighestLowest:
            plt.savefig(direc+f"LDOS_{Lattice.defect_type}_{side_length}_all.png")  
        else:
            plt.savefig(direc+f"LDOS_clip_{Lattice.defect_type}_{side_length}_all.png")

    # -------------------------------------
    elif plot_defect_type in ["none", "vacancy"]:
        Lattice = DefectSquareLattice(side_length, plot_defect_type)

        M_back_vals = [-2.5, -1.0, 1.0, 2.5]
        fig, axs = plt.subplots(4, 2, figsize=(8, 12))
        spectrum_axs = axs[:, 0].flatten()
        ldos_axs = axs[:, 1].flatten()
        for spectrum_ax, ldos_ax, M_background in zip(spectrum_axs, ldos_axs, M_back_vals):
            hamiltonian = Lattice.compute_hamiltonian(M_background, None)
            LDOS, eigenvalues, gap = Lattice.compute_LDOS(hamiltonian, number_of_states=2, returnEigenvalues=True, returnGap=True)
            ldos_ax.scatter(Lattice.X, Lattice.Y, c=LDOS, s=10, cmap='inferno')
            ldos_ax.spines['top'].set_visible(True)
            ldos_ax.spines['right'].set_visible(True)
            ldos_ax.spines['bottom'].set_visible(True)
            ldos_ax.spines['left'].set_visible(True)
            ldos_ax.set_xticks([])
            ldos_ax.set_yticks([])
            ldos_ax.set_aspect('equal')
            ldos_ax.set_title(rf"$m_0^{{back}}$ = {M_background}")

            spectrum_ax.scatter(np.arange(len(eigenvalues)), eigenvalues, label="Gap = {:.2e}".format(gap))
            spectrum_ax.set_xlabel("Eigenvalue Index")
            spectrum_ax.set_ylabel("Energy")
            spectrum_ax.set_title(f"Spectrum")
            spectrum_ax.legend()
            #spectrum_ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

            if ldos_ax.collections:  # Check if collections exist
                norm = plt.Normalize(vmin=max(1e-10, min(coll.get_array().min() for coll in ldos_ax.collections)),
                                        vmax=max(coll.get_array().max() for coll in ldos_ax.collections))
                for coll in ldos_ax.collections:
                    coll.set_norm(norm)
                
            cbar = fig.colorbar(ldos_ax.collections[0], ax=ldos_ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label("LDOS")
            cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))

        formatter = ticker.FormatStrFormatter('%.1e')
        for ax in axs.flatten():
            for im in ax.collections:
                if hasattr(im, 'colorbar') and im.colorbar is not None:
                    im.colorbar.ax.yaxis.set_major_formatter(formatter)

        fig.suptitle(f"{Lattice.defect_type.capitalize()}", fontsize=20)
        direc="./Defects/Plots/LDOS/"
        plt.tight_layout()
        plt.show()
        #plt.savefig(direc+f"LDOS_{Lattice.defect_type}.png")

    # -------------------------------------
    elif plot_defect_type == "frenkel_pair":
        M_back_vals = [-2.5, -1.0, 1.0, 2.5]
        M_sub_vals = [-2.5, -1.0, 1.0, 2.5]
        n_rows, n_cols = 4, 8
        scale = 4
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(scale*n_cols, scale*n_rows))   
        for i, M_background in enumerate(M_back_vals):  
            spectrum_axs = axs[:, 0+2*i].flatten()
            ldos_axs = axs[:, 1+2*i].flatten()
            for spectrum_ax, ldos_ax, M_substitution in zip(spectrum_axs, ldos_axs, M_sub_vals):
                all_LDOS = []
                all_x = []
                all_y = []
                all_eigenvalues = []
                all_gap = []
                for frenkel_pair_index in range(8):
                    Lattice = DefectSquareLattice(side_length, plot_defect_type, frenkel_pair_index=frenkel_pair_index)
                    hamiltonian = Lattice.compute_hamiltonian(M_background, M_substitution)
                    LDOS, eigenvalues, gap = Lattice.compute_LDOS(hamiltonian, number_of_states=2, returnEigenvalues=True, returnGap=True)
                    all_LDOS.append(LDOS)
                    all_x.append(Lattice.X)
                    all_y.append(Lattice.Y)
                    all_eigenvalues.append(eigenvalues)
                    all_gap.append(gap)
                
                all_x = np.concatenate(all_x)
                all_y = np.concatenate(all_y)
                all_LDOS = np.concatenate(all_LDOS)
                # Find unique (X, Y) pairs and sum LDOS for each unique site
                coords = np.stack((all_x, all_y), axis=1)
                unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
                summed_LDOS = np.zeros(len(unique_coords))
                np.add.at(summed_LDOS, inverse_indices, all_LDOS)
                summed_LDOS /= 8.0

                if clipHighestLowest:
                    summed_LDOS = np.clip(summed_LDOS, np.sort(summed_LDOS)[1], np.sort(summed_LDOS)[-2])
                ldos_ax.scatter(unique_coords[:, 0], unique_coords[:, 1], c=summed_LDOS, s=50, cmap='inferno')
                ldos_ax = format_ldos_ax(ldos_ax, rf"$m_0^{{sub}}$ = {M_substitution}")
    
                spectrum_ax.scatter(np.arange(len(eigenvalues)), eigenvalues, label="Gap = {:.2e}".format(gap))
                spectrum_ax.set_xlabel("Eigenvalue Index")
                spectrum_ax.set_ylabel("Energy")
                spectrum_ax.set_title(f"Spectrum")
                #spectrum_ax.legend()
                #spectrum_ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

                if ldos_ax.collections:  # Check if collections exist
                    norm = plt.Normalize(vmin=max(1e-10, min(coll.get_array().min() for coll in ldos_ax.collections)),
                                            vmax=max(coll.get_array().max() for coll in ldos_ax.collections))
                    for coll in ldos_ax.collections:
                        coll.set_norm(norm)
                    
                cbar = fig.colorbar(ldos_ax.collections[0], ax=ldos_ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label("LDOS")
                cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))

            fig.text((2*i+1)/8, 0.95, rf"$m_0^{{back}}$ = {M_background}", fontsize=20, ha='center')



        formatter = ticker.FormatStrFormatter('%.1e')
        for ax in axs.flatten():
            for im in ax.collections:
                if hasattr(im, 'colorbar') and im.colorbar is not None:
                    im.colorbar.ax.yaxis.set_major_formatter(formatter)

        if not clipHighestLowest:
            fig.suptitle("Frenkel Pair", fontsize=30)
        else:
            fig.suptitle("Frenkel Pair (clipped)", fontsize=30)
        plt.tight_layout()
        
        direc="./Defects/Plots/LDOS/"
        if not clipHighestLowest:
            plt.savefig(direc+f"LDOS_{Lattice.defect_type}_{side_length}_all.png")  
            pass
        else:
            plt.savefig(direc+f"LDOS_clip_{Lattice.defect_type}_{side_length}_all.png")
            pass


def probe_point():
    Lattice = DefectSquareLattice(14, "interstitial")
    dx, dy = Lattice.dx, Lattice.dy

    theta = np.arctan2(dy, dx) % (2 * np.pi)
    distance_mask = np.maximum(np.abs(dx), np.abs(dy)) <= 1

    principal_mask = (((dx == 0) & (dy != 0)) | ((dx != 0) & (dy == 0))) & (distance_mask)
    diagonal_mask  = ((np.abs(dx) == np.abs(dy)) & ((dx != 0) & (dy != 0))) & (distance_mask)
    all_mask = principal_mask | diagonal_mask

    d_r = np.where(all_mask, np.sqrt(dx**2 + dy**2), 0.0 + 0.0j)
    F_p = np.where(principal_mask, np.exp(1  - d_r), 0. + 0.j)
    d_cos = np.where(all_mask, np.cos(theta), 0. + 0.j)
    d_sin = np.where(all_mask, np.sin(theta), 0. + 0.j)

    X, Y = Lattice.X, Lattice.Y

    plt.scatter(X, Y, c=diagonal_mask[np.max(Lattice.lattice)//2+7].real, s=50, cmap='inferno')
    plt.colorbar()
    plt.show()


    def OLD_plot_spectrum_ldos(self, m_background_values:"list[float]" = [2.5, 1.0, -1.0, -2.5], 
                             m_substitution_values:"list[float] | None" = None, doLargeDefectFigure:bool = False, number_of_states:int = 2): 
        def plot_ldos_ax(ldos_ax:plt.Axes, LDOS, X, Y):
            # Dynamically set marker size based on number of points and axes size
            bbox = ldos_ax.get_window_extent().transformed(ldos_ax.figure.dpi_scale_trans.inverted())
            width, height = bbox.width * ldos_ax.figure.dpi, bbox.height * ldos_ax.figure.dpi
            area = width * height
            N = len(X)
            # Heuristic: marker area is a fraction of axes area divided by number of points
            marker_area = max(area / (N * 10), 0)
            scat = ldos_ax.scatter(X, Y, c=LDOS, s=marker_area, cmap='jet')
            ldos_ax.set_xticks([np.min(X), (np.max(X) + np.min(X)) / 2, np.max(X)])
            ldos_ax.set_yticks([np.min(X), (np.max(X) + np.min(X)) / 2, np.max(X)])
            tick_labels = [np.min(X) + 1, (np.max(X) + np.min(X)) // 2 + 1, np.max(X) + 1]
            ldos_ax.set_xticklabels([str(int(label)) for label in tick_labels], fontsize=20)
            ldos_ax.set_yticklabels([str(int(label)) for label in tick_labels], fontsize=20)
            ldos_ax.set_xlabel(r"$x$", fontsize=20)
            ldos_ax.set_ylabel(r"$y$", fontsize=20)
            ldos_ax.set_aspect('equal')
            return ldos_ax
        
        def plot_spectrum_ax(spectrum_ax:plt.Axes, eigenvalues:np.ndarray, scatter_label:str, ldos_idxs:np.ndarray):
            x_values = np.arange(len(eigenvalues))
            idxs_mask = np.isin(x_values, ldos_idxs)
            spectrum_ax.scatter(x_values[~idxs_mask], eigenvalues[~idxs_mask], s=25, color = 'black', zorder = 0)
            spectrum_ax.scatter(x_values[ idxs_mask], eigenvalues[ idxs_mask], s=25, color = 'red',   zorder = 1)
            spectrum_ax.set_xticks([])
            spectrum_ax.set_xlabel(r"$n$", fontsize=20)
            spectrum_ax.set_ylabel(r"$E_n$", fontsize=20)
            spectrum_ax.tick_params(axis='y', labelsize=20)
            spectrum_ax.annotate(
                scatter_label,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                ha='left',
                va='top',
                fontsize=16,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.0)
            )

        if m_substitution_values is None:
            m_substitution_values = np.array(m_background_values).copy()

        # Get shape of the figure based on the defect type
        if self.defect_type in ["none", "vacancy"]:
            m_substitution_values = [None] if doLargeDefectFigure is False else [None] * 2
            n_cols, n_rows = 2 * len(m_background_values), len(m_substitution_values)
        else:
            n_cols, n_rows = 2 * len(m_background_values), len(m_substitution_values) - 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

        if n_rows == 1:
            axs = np.array([axs])

        for i, m_background in enumerate(m_background_values):
            spectrum_axs = axs[:, 0 + 2 * i].flatten()
            ldos_axs = axs[:, 1 + 2 * i].flatten()
            
            good_m_sub_vals = np.array(m_substitution_values)[np.array(m_substitution_values) != m_background]

            for j, (spectrum_ax, ldos_ax, m_substitution) in enumerate(zip(spectrum_axs, ldos_axs, good_m_sub_vals)):
                if m_substitution == m_background:
                    continue
                
                if j == 1 and doLargeDefectFigure and self.defect_type in ["none", "vacancy"]:
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self.LargeDefectLattice._compute_for_figure(m_background, m_substitution, number_of_states)
                elif doLargeDefectFigure and self.defect_type not in ["none", "vacancy"]:
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self.LargeDefectLattice._compute_for_figure(m_background, m_substitution, number_of_states)
                else:
                    LDOS, eigenvalues, gap, bott_index, X, Y, ldos_idxs = self._compute_for_figure(m_background, m_substitution, number_of_states)
                
                LDOS -= np.min(LDOS)
                if np.max(LDOS) > 0:
                    LDOS /= np.max(LDOS)
                
                plot_ldos_ax(ldos_ax, LDOS, X, Y)
                if self.defect_type in ["none", "vacancy"]:
                    param_name = r"$m_0=$"+f"{m_background}"
                elif self.defect_type in ["substitution"]:
                    param_name = f"$m_0^{{\\text{{sub}}}}=$"+f"{m_substitution}"
                else:
                    param_name = f"$m_0^{{\\text{{int}}}}=$"+f"{m_substitution}"

                plot_spectrum_ax(spectrum_ax, eigenvalues, f"Gap = {gap:.2f}\nBott Index = {bott_index}\n"+param_name, ldos_idxs)

                if False:
                    if self.defect_type not in ["none", "vacancy"]:
                        spectrum_ax.annotate(f"$m_0^{{sub}}$ = {m_substitution}", xy=(-0.25, 0.5), xycoords='axes fraction', ha='center', fontsize=12, rotation=90, va='center')
                    else:
                        spectrum_ax.annotate(f"$m_0^{{back}}$ = {m_background}", xy=(-0.25, 0.5), xycoords='axes fraction', ha='center', fontsize=12, rotation=90, va='center')

                cbar = fig.colorbar(ldos_ax.collections[0], ax=ldos_ax, orientation='vertical', fraction=0.046, pad=0.04)
                cbar.ax.yaxis.set_ticks([0.0, 1.0])
                cbar.ax.tick_params(labelsize=20)
                cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.f'))

        
        if n_rows == 1:
            plt.subplots_adjust(top=0.8)
        else:
            plt.subplots_adjust(top=0.9)
        set_labels = [f"({lab})" for lab in "abcdefghijklmnopqrstuvwxyz"[:len(m_background_values)]]
        for i, m_background in enumerate(m_background_values):
            if n_rows == 1:
                fig.text((2*i+1)/(2 * len(m_background_values)), 0.85, set_labels[i], fontsize=36, ha='center')
            else:
                fig.text((2*i+1)/(2 * len(m_background_values)), 0.95, set_labels[i], fontsize=36, ha='center')
        plt.tight_layout()
        return fig, axs



