% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/data_simulation.R
\name{simulate_shapes}
\alias{simulate_shapes}
\title{Data Simulation for CNN}
\usage{
simulate_shapes(n, size, p = 0.5)
}
\arguments{
\item{n}{number of images}

\item{size}{size of the (quadratic) images}

\item{p}{probability of generating a rectangle (1-p for ellipsoids)}
}
\value{
array of dimension (n, 1, size, size)
}
\description{
generates images of rectangles and ellipsoids
}
\details{
This function generates simple data to demonstrate the usage of cnn().
The generated images are of centered rectangles and ellipsoids with random widths and heights.
}
\author{
Armin Schenk
}
