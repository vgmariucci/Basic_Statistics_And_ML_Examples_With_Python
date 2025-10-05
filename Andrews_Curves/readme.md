# Andrews Curves

Andrews curves are a visualization technique for multivariate data, invented by statistician David Andrews in 1972. They transform each multivariate observation into a smooth curve, allowing you to visualize high-dimensional data in two dimensions.

### How They Work

For a data point with features (x₁, x₂, x₃, ..., xₙ), Andrews curves map it to a function:

    f(t) = x₁/√2 + x₂·sin(t) + x₃·cos(t) + x₄·sin(2t) + x₅·cos(2t) + ...

Each observation becomes a curve plotted over a range of t values (typically -π to π). 

Observations with similar features produce similar curves, making it useful for:

- **Detecting clusters**: similar observations cluster together visually

- **Identifying outliers**: unusual observations produce 
distinctive curves

- **Comparing groups**:  different classes often show different curve patterns

### Applications

- **Classification problems**: Visualizing separation between classes (Iris dataset, cancer detection)

- **Quality control**: Identifying defective products in manufactoring

- **Financial analysis**: Comparing companies based on multiple financial metrics

- **Anomaly detection**: Spotting unusual patterns in sensor data or network traffic

### Key Points to Remember

- **Standardize you data**: Features with different scales can dominate the curves

- **Best for 3-10 features**: Too few and you don't neet it; too many and it becomes cluttered

- **Keep observations moderate**: Works best with 50-500 observations per plot

- **Color by class**: Always use the class/category parameter to distinguish groups


The main advantage of Andrews curves over other techiniques like parallel coordinates is that observations with similar feature values will have curves close together, making it easier to spot natural clusters in your data.