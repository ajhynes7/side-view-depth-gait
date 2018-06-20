import functools
import numpy as np

import modules.math_funcs as mf


class MeanShift:

    def cluster(points, kernel='flat', radius=1, eps=0.01):
        """
        Cluster points with the mean shift algorithm.

        Parameters
        ----------
        points : ndarray
            (n, d) array of n points with dimension d.
        kernel : str, optional
            Kernel (default 'flat') used for weighting points when shifting
            the mean position. Options are {'flat', 'gaussian'}.
        radius : {int, float}, optional
            Radius for mean shift (default 1).
        eps : {int, float}, optional
            The convergence criterion epsilon (default 0.01).
            The shifting process terminates when the new mean position has
            moved by a distance less than eps since the previous iteration.

        Returns
        -------
        labels : ndarray
            Cluster label of each point.
        centroids : ndarray
            (k, d) array of k cluster centroids with dimension d.
        k : int
            Number of clusters.

        Examples
        --------
        >>> points = np.array([[1, 2], [2, 3], [20, 3]])
        >>> labels, centroids, k = MeanShift.cluster(points, radius=2)

        >>> labels
        array([1, 1, 0])

        >>> centroids
        array([[20. ,  3. ],
               [ 1.5,  2.5]])

        >>> k
        2

        """
        n_points, n_dimensions = points.shape

        index_matrix = np.full((n_points, n_points), False)
        all_centroids = np.full((n_points, n_dimensions), np.nan)

        for i, mean_pos in enumerate(points):

            # Shift mean until convergence
            mean_pos, in_radius = MeanShift._shift_to_convergence(
                points, mean_pos, kernel, radius, eps)

            index_matrix[i, :] = in_radius
            all_centroids[i, :] = mean_pos

        _, unique_indices, labels = np.unique(
            index_matrix, return_index=True, return_inverse=True, axis=0)

        centroids = all_centroids[unique_indices, :]
        k = len(centroids)

        return labels, centroids, k

    def _shift_to_convergence(points, mean_pos, kernel, radius, eps):
        """
        Shift a mean position until it converges to a final position.

        Parameters
        ----------
        points : ndarray
            (n, d) array of n points with dimension d.
        mean_pos : ndarray
            Initial mean position that will be shifted.
        kernel : str, optional
            Kernel (default 'flat') used for weighting points when shifting
            the mean position. Options are {'flat', 'gaussian'}.
        radius : {int, float}, optional
            Radius for mean shift (default 1).
        eps : {int, float}, optional
            The convergence criterion epsilon (default 0.01).
            The shifting process terminates when the new mean position has
            moved by a distance less than eps since the previous iteration.

        Returns
        -------
        mean_pos : ndarray
            Final mean position.
        in_radius : ndarray
            (n, ) array of booleans. Element i is true if point i is within
            the radius of the final mean position.

        Examples
        --------
        >>> points = np.array([[1, 2], [2, 3], [20, 3]])
        >>> mean_pos = np.array([2, 3])

        >>> MeanShift._shift_to_convergence(points, mean_pos, 'flat', 5, 0.01)
        (array([1.5, 2.5]), array([ True,  True, False]))

        """

        # Create function with specified kernel type
        kernel_func = functools.partial(MeanShift._kernel_function,
                                        kernel=kernel)

        while True:

            prev_mean_pos = mean_pos

            distances = np.linalg.norm(points - mean_pos, axis=1)

            # Masses defined by kernel function
            masses = kernel_func(distances, radius)

            # A weighted mean of the points can be obtained by calculating the
            # centre of mass
            mean_pos = mf.centre_of_mass(points, masses)

            if np.linalg.norm(prev_mean_pos - mean_pos) < eps:
                # Mean position has converged

                in_radius = distances <= radius
                break

        return mean_pos, in_radius

    def _kernel_function(distances, radius, kernel):
        """
        Return the mass of each point.
        Used for calculating a new mean position by finding the centre of mass.

        The mass of each point depends on the type of kernel used.

        Parameters
        ----------
        distances : ndarray
            (n, ) array of distances.
            Distance of each point to the mean position.
        radius : {int, float}, optional
            Radius for mean shift (default 1).
        kernel : str, optional
            Kernel (default 'flat') used for weighting points when shifting
            the mean position. Options are {'flat', 'gaussian'}.

        Returns
        -------
        masses : ndarray
            (n, ) array of masses.

        Raises
        ------
        ValueError
            When kernel string is not one of the accepted options.

        Examples
        --------
        >>> distances = np.array([1, 10, 2, 5])
        >>> radius = 5

        >>> MeanShift._kernel_function(distances, radius, 'flat')
        array([1., 0., 1., 1.])

        >>> masses = MeanShift._kernel_function(distances, radius, 'gaussian')
        >>> np.round(masses, 2)
        array([0.08, 0.01, 0.07, 0.05])

        >>> MeanShift._kernel_function(distances, radius, 'k')
        Traceback (most recent call last):
        ValueError: Invalid kernel type

        """
        if kernel == 'flat':
            masses = np.zeros(len(distances))
            masses[distances <= radius] = 1

        elif kernel == 'gaussian':
            masses = mf.gaussian(distances, sigma=radius)

        else:
            raise ValueError('Invalid kernel type')

        return masses


if __name__ == "__main__":

    import doctest
    doctest.testmod()
