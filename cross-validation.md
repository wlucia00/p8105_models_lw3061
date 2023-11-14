cross validation
================
Lucia Wang
2023-11-14

### notes

model selection is difficult

- nested models have tests
- non-nested do not - balance goodness of fit with “complexity”

questions: is model complex enough? too complex? under/over fitting?
high bias? variances? *will my model generalize to future datasets*?

cross validation - split data into training and testing

- training data to build your model
- testing data to evaluate out-of-sample fit
- ratio depends on data size (jeff likes 80/20)
- evaluate with root mean squared error

refinements and variations

- folding partitions data into equal subsets - one for testing, other
  for training, alternate
- repeat the process - variability in accuracy, find differences in
  models across splits

cross validation is general

- compare candidates that are “traditional” (lasso)
- modern methods? automated variable selection, additive models,
  regression trees

## nonlinear data and cv

``` r
nonlin_df = 
  tibble(
    id = 1:100,
    x = runif(100, 0, 1),
    y = 1 - 10 * (x - .3) ^ 2 + rnorm(100, 0, .3)
  )

nonlin_df |> 
  ggplot(aes(x = x, y = y)) + 
  geom_point()
```

![](cross-validation_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

Do the train/test split

``` r
train_df = sample_n(nonlin_df, 80)
test_df = anti_join(nonlin_df, train_df, by="id")

train_df |> ggplot(aes(x=x, y=y)) +
  geom_point() +
  geom_point(data=test_df, color="red")
```

![](cross-validation_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->
about 20 points randomly selected to be in the testing df. there will be
noise (residual error) but if one model is better to predict the red
points after being fit to the black points, maybe that is the best
model.

``` r
linear_model = lm(y~x, data=train_df)

# quick visualization
train_df |> 
  modelr::add_predictions(linear_model) |>
  ggplot(aes(x=x, y=y)) +
  geom_point() +
  geom_line(aes(y=pred))
```

![](cross-validation_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->
this model is not that good, the line doesnt really hit any of the
points.

use `mgcv::gam` instead!

``` r
smooth_mod = mgcv::gam(y~s(x), data=train_df)
wiggly = mgcv::gam(y~s(x, k=30), sp=10e-6, data=train_df)

train_df |> 
  add_predictions(smooth_mod) |> 
  ggplot(aes(x = x, y = y)) + geom_point() + 
  geom_line(aes(y = pred), color = "red")
```

![](cross-validation_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
train_df |> 
  add_predictions(wiggly) |> 
  ggplot(aes(x = x, y = y)) + geom_point() + 
  geom_line(aes(y = pred), color = "red")
```

![](cross-validation_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

lets look at the RMSE

``` r
modelr::rmse(linear_model, train_df)
```

    ## [1] 0.7178747

``` r
modelr::rmse(smooth_mod, train_df)
```

    ## [1] 0.2874834

``` r
modelr::rmse(wiggly, train_df)
```

    ## [1] 0.2498309

the wiggly one seems to be doing the best. what about on the `test_df`?

``` r
modelr::rmse(linear_model, test_df)
```

    ## [1] 0.7052956

``` r
modelr::rmse(smooth_mod, test_df)
```

    ## [1] 0.2221774

``` r
modelr::rmse(wiggly, test_df)
```

    ## [1] 0.289051

the smooth model seems to do better here.
