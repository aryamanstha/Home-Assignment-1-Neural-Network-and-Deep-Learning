# Home Assignment 1: Neural Network and Deep Learning 

**Name:** Aryaman Shrestha 
**Student ID:** 700788013 

## Task 1: Tensor Manipulations & Reshaping

**Objective:** Understand tensor operations and broadcasting in TensorFlow.

**Key Concepts:**
- **Tensor Creation:** Generated random tensors with shape (4, 6) using `tf.random.uniform()`
- **Shape & Rank:** Retrieved tensor dimensions and rank using TensorFlow functions
- **Reshaping:** Transformed the tensor from (4, 6) → (2, 3, 4) using `tf.reshape()`
- **Transposing:** Rearranged dimensions from (2, 3, 4) → (3, 2, 4) using `tf.transpose()` with permutation [1, 0, 2]
- **Broadcasting:** Demonstrated how a smaller tensor (1, 4) can be broadcast to match larger tensor shapes in arithmetic operations

**Broadcasting Explanation:**
TensorFlow broadcasting allows tensors with different shapes to work together seamlessly. The key rules are:
1. Dimensions are compared from right to left
2. Compatible dimensions are either equal or one of them is 1
3. Dimensions of size 1 are expanded to match the other tensor
4. Missing leading dimensions are treated as size 1

This enables efficient operations without explicit replication of data.

---

## Task 2: Loss Functions & Hyperparameter Tuning

**Objective:** Compare different loss functions and understand their behavior with model predictions.

**Implementations:**
- **Mean Squared Error (MSE):** Measures the average squared difference between predictions and true values
- **Categorical Cross-Entropy (CCE):** Evaluates probability distribution differences, ideal for multi-class classification

**Experiment:**
- Defined true values (one-hot encoded) and two sets of predictions (original and improved)
- Calculated loss values for both functions on each prediction set
- Visualized results using a bar chart showing how loss decreases when predictions improve

**Key Findings:**
- Both loss functions decrease when predictions become more accurate
- CCE is more suitable for classification tasks with probability outputs
- MSE can be used for regression or classification but is less interpretable for probabilities

---

## Task 3: Training Models with Different Optimizers

**Objective:** Compare how Adam and SGD optimizers affect MNIST classification performance.

**Dataset & Preprocessing:**
- Loaded MNIST (60,000 training, 10,000 test images of handwritten digits)
- Normalized pixel values to [0, 1] range by dividing by 255

**Model Architecture:**
A Convolutional Neural Network (CNN) with:
- Input layer: 28×28×1 images
- Conv2D layer: 32 filters, 3×3 kernel, ReLU activation
- MaxPooling2D: 2×2 pooling
- Flatten layer
- Dense layer: 128 units with ReLU activation
- Output layer: 10 units with softmax activation (one per digit class)

**Training & Comparison:**
- **Adam Optimizer:** Adaptive learning rate, generally converges faster and more reliably
- **SGD Optimizer:** Standard gradient descent, can be slower but sometimes generalizes better

**Results:**
- Both models trained for 20 epochs with 20% validation split
- Plotted training and validation accuracy curves for direct comparison
- Adam typically achieves higher accuracy and faster convergence

---

## Task 4: Training with TensorBoard Logging & Overfitting Detection

**Objective:** Train a neural network while monitoring metrics in real-time using TensorBoard.

**Model Architecture:**
A simple feedforward neural network:
- Input: 28×28 MNIST images
- Flatten layer
- Hidden Dense layer: 128 units with ReLU activation
- Output Dense layer: 10 units with softmax activation

**TensorBoard Integration:**
- Used `tf.keras.callbacks.TensorBoard()` to log training metrics
- Logs are stored in `logs/fit/` with timestamp-based subdirectories
- Records training/validation loss and accuracy for each epoch

**Training Results & Analysis:**

1. **Overfitting Patterns Observed:**
   - Training accuracy steadily increases to ~98.47% by epoch 5
   - Validation accuracy rises initially (~95.5% → 97.25%) but drops to ~96.95% in the final epoch
   - Growing gap between training and validation metrics after epoch 3 indicates overfitting

2. **Detecting Overfitting with TensorBoard:**
   - Divergence between training and validation accuracy curves signals overfitting
   - Training loss continues decreasing while validation loss increases
   - Optimal stopping point is typically when validation metrics plateau

3. **Effect of Increased Epochs:**
   - More epochs allow the model to learn training data patterns more thoroughly
   - Beyond the optimal point, further training causes the model to overfit
   - Training accuracy approaches 100% while validation accuracy stagnates or decreases
   - Ideal training stops when validation performance stops improving

**Test Performance:**
- Model evaluated on held-out test set for unbiased accuracy assessment

---

## Key Takeaways

1. **Tensor Operations:** TensorFlow provides efficient tools for manipulating multi-dimensional arrays with broadcasting support
2. **Loss Functions:** Choice of loss function depends on the problem type and prediction format
3. **Optimizers:** Different optimizers have trade-offs between convergence speed and generalization
4. **Monitoring:** TensorBoard is essential for detecting overfitting and optimizing training duration
5. **Neural Networks:** Careful architecture design and early stopping strategies prevent overfitting


