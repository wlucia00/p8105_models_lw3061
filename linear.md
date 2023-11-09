Linear
================
Lucia Wang
2023-11-09

## Linear models

Import + tidy nyc airbnb data

``` r
data("nyc_airbnb") 

nyc_airbnb = nyc_airbnb |>
  mutate(stars = review_scores_location / 2) |>
  select(
    price, stars, borough=neighbourhood_group, neighbourhood, room_type
  ) |>
  filter(borough != "Staten Island")
```

Let’s fit a model

``` r
fit = 
  nyc_airbnb |>
  lm(price ~ stars + borough, data= _) # since we piped it in, we use data= _ 
```

Let’s look at it! `broom` package has lots of functions for summary or
cleaning table

``` r
fit
```

    ## 
    ## Call:
    ## lm(formula = price ~ stars + borough, data = nyc_airbnb)
    ## 
    ## Coefficients:
    ##      (Intercept)             stars   boroughBrooklyn  boroughManhattan  
    ##           -70.41             31.99             40.50             90.25  
    ##    boroughQueens  
    ##            13.21

``` r
summary(fit) # matrix
```

    ## 
    ## Call:
    ## lm(formula = price ~ stars + borough, data = nyc_airbnb)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -169.8  -64.0  -29.0   20.2 9870.0 
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)       -70.414     14.021  -5.022 5.14e-07 ***
    ## stars              31.990      2.527  12.657  < 2e-16 ***
    ## boroughBrooklyn    40.500      8.559   4.732 2.23e-06 ***
    ## boroughManhattan   90.254      8.567  10.534  < 2e-16 ***
    ## boroughQueens      13.206      9.065   1.457    0.145    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 181.5 on 30525 degrees of freedom
    ##   (9962 observations deleted due to missingness)
    ## Multiple R-squared:  0.03423,    Adjusted R-squared:  0.03411 
    ## F-statistic: 270.5 on 4 and 30525 DF,  p-value: < 2.2e-16

``` r
summary(fit)$coef # look at just coeffs
```

    ##                   Estimate Std. Error   t value     Pr(>|t|)
    ## (Intercept)      -70.41446  14.020697 -5.022180 5.137589e-07
    ## stars             31.98989   2.527500 12.656733 1.269392e-36
    ## boroughBrooklyn   40.50030   8.558724  4.732049 2.232595e-06
    ## boroughManhattan  90.25393   8.567490 10.534465 6.638618e-26
    ## boroughQueens     13.20617   9.064879  1.456850 1.451682e-01

``` r
coef(fit) # pulls out that table
```

    ##      (Intercept)            stars  boroughBrooklyn boroughManhattan 
    ##        -70.41446         31.98989         40.50030         90.25393 
    ##    boroughQueens 
    ##         13.20617

``` r
# fitted.values(fit) # all the fitted values - big table

fit |> broom::glance()
```

    ## # A tibble: 1 × 12
    ##   r.squared adj.r.squared sigma statistic   p.value    df   logLik    AIC    BIC
    ##       <dbl>         <dbl> <dbl>     <dbl>     <dbl> <dbl>    <dbl>  <dbl>  <dbl>
    ## 1    0.0342        0.0341  182.      271. 6.73e-229     4 -202113. 4.04e5 4.04e5
    ## # ℹ 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>

``` r
fit |> broom::tidy() #gives a tibble
```

    ## # A tibble: 5 × 5
    ##   term             estimate std.error statistic  p.value
    ##   <chr>               <dbl>     <dbl>     <dbl>    <dbl>
    ## 1 (Intercept)         -70.4     14.0      -5.02 5.14e- 7
    ## 2 stars                32.0      2.53     12.7  1.27e-36
    ## 3 boroughBrooklyn      40.5      8.56      4.73 2.23e- 6
    ## 4 boroughManhattan     90.3      8.57     10.5  6.64e-26
    ## 5 boroughQueens        13.2      9.06      1.46 1.45e- 1

``` r
fit |> broom::tidy() |>
  mutate(term= str_replace(term, "^borough", "Borough: ")) |>
  select(term, estimate, p.value) |>
  knitr::kable(digits=3)
```

| term               | estimate | p.value |
|:-------------------|---------:|--------:|
| (Intercept)        |  -70.414 |   0.000 |
| stars              |   31.990 |   0.000 |
| Borough: Brooklyn  |   40.500 |   0.000 |
| Borough: Manhattan |   90.254 |   0.000 |
| Borough: Queens    |   13.206 |   0.145 |

## Diagnostics

you can add residuals or fitted values to the df

``` r
nyc_airbnb |>
  modelr::add_residuals(fit)
```

    ## # A tibble: 40,492 × 6
    ##    price stars borough neighbourhood room_type        resid
    ##    <dbl> <dbl> <chr>   <chr>         <chr>            <dbl>
    ##  1    99   5   Bronx   City Island   Private room      9.47
    ##  2   200  NA   Bronx   City Island   Private room     NA   
    ##  3   300  NA   Bronx   City Island   Entire home/apt  NA   
    ##  4   125   5   Bronx   City Island   Entire home/apt  35.5 
    ##  5    69   5   Bronx   City Island   Private room    -20.5 
    ##  6   125   5   Bronx   City Island   Entire home/apt  35.5 
    ##  7    85   5   Bronx   City Island   Entire home/apt  -4.53
    ##  8    39   4.5 Bronx   Allerton      Private room    -34.5 
    ##  9    95   5   Bronx   Allerton      Entire home/apt   5.47
    ## 10   125   4.5 Bronx   Allerton      Entire home/apt  51.5 
    ## # ℹ 40,482 more rows

``` r
nyc_airbnb |> modelr::add_predictions(fit)
```

    ## # A tibble: 40,492 × 6
    ##    price stars borough neighbourhood room_type        pred
    ##    <dbl> <dbl> <chr>   <chr>         <chr>           <dbl>
    ##  1    99   5   Bronx   City Island   Private room     89.5
    ##  2   200  NA   Bronx   City Island   Private room     NA  
    ##  3   300  NA   Bronx   City Island   Entire home/apt  NA  
    ##  4   125   5   Bronx   City Island   Entire home/apt  89.5
    ##  5    69   5   Bronx   City Island   Private room     89.5
    ##  6   125   5   Bronx   City Island   Entire home/apt  89.5
    ##  7    85   5   Bronx   City Island   Entire home/apt  89.5
    ##  8    39   4.5 Bronx   Allerton      Private room     73.5
    ##  9    95   5   Bronx   Allerton      Entire home/apt  89.5
    ## 10   125   4.5 Bronx   Allerton      Entire home/apt  73.5
    ## # ℹ 40,482 more rows

``` r
nyc_airbnb |> 
  modelr::add_residuals(fit) |> 
  ggplot(aes(x = borough, y = resid)) + geom_violin()
```

![](linear_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

## Anova

test means between groups

``` r
fit_null = lm(price ~ stars + borough, data = nyc_airbnb)
fit_alt = lm(price ~ stars + borough + room_type, data = nyc_airbnb)

anova(fit_null, fit_alt) |> 
  broom::tidy()
```

    ## # A tibble: 2 × 7
    ##   term                        df.residual    rss    df   sumsq statistic p.value
    ##   <chr>                             <dbl>  <dbl> <dbl>   <dbl>     <dbl>   <dbl>
    ## 1 price ~ stars + borough           30525 1.01e9    NA NA            NA       NA
    ## 2 price ~ stars + borough + …       30523 9.21e8     2  8.42e7     1394.       0

what about borough level differences (interaction)

``` r
fit = 
  nyc_airbnb |>
  lm(price ~ stars*borough + room_type*borough, data= _)
```

This works but takes a while/ hard to understand. Instead, nest within
the boroughs

``` r
airbnb_lm = function(df) {
  lm(price ~ stars + room_type, data=df)
}

nyc_airbnb |>
  nest(df= -borough) |>
  mutate(models = map(df, airbnb_lm),
         results = map(models, broom::tidy))  |>
  select(borough, results) |>
  unnest(results) |>
  select(borough, term, estimate) |>
  pivot_wider(names_from = term,
              values_from = estimate) |>
  knitr::kable(digits=2)
```

| borough   | (Intercept) | stars | room_typePrivate room | room_typeShared room |
|:----------|------------:|------:|----------------------:|---------------------:|
| Bronx     |       90.07 |  4.45 |                -52.91 |               -70.55 |
| Queens    |       91.58 |  9.65 |                -69.26 |               -94.97 |
| Brooklyn  |       69.63 | 20.97 |                -92.22 |              -105.84 |
| Manhattan |       95.69 | 27.11 |               -124.19 |              -153.64 |

you can create an anonymous function too using`\(df)` then defining it
there

## Binary outcomes

using Wapo homicides data
