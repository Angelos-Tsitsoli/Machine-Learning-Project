# Assignment #1
## Calculation of derivatives
Let $\( f: \mathbb{R}^d \to \mathbb{R} \)$ with $\( f(x) = (x - a)^\top A (x - a) \)$, where $\( A \in \mathbb{R}^{d \times d} \)$ is a symmetrical matrix $( \( A^\top = A \))$ and $\( a \in \mathbb{R}^d \)$. The gradient of f is calculated and also calculate the total minimum of $\min_{\mathbf{X}} \| \mathbf{A} - \mathbf{X}\mathbf{B} \|_F^2$

## Gradient Descent

$f_1(x; w) = \left( 1 + e^{-(w_0 + w_1 x)} \right)^{-1}, w \in \mathbb{R}^2$



$f_2(x; w) = w_0 + w_1 \sigma(w_2 + w_3 x), w \in \mathbb{R}^4$


$L_1(w) = - \frac{1}{n} \sum_{i=1}^n \left( y_i \log_2 f_1(x_i; w) + (1 - y_i) \log_2 (1 - f_1(x_i; w)) \right) $


$L_2(w) = \frac{1}{n} \sum_{i=1}^n \left( y_i - f_2(x_i; w) \right)^2 $

$$
\sigma(x) = 
\begin{cases} 
  x & \text{if } x > 0 \\
  0 & \text{otherwise}
\end{cases}
$$


Implementation in Python and running of the Gradient Descent algorithm for some cases with the above information.


## Face recognition(using Eigenfaces)
Implementation of the algorithm in Python also using the images from faces directory. After the implementation solve different cases.


# Assignment #2

## Feedforward Neural Network



