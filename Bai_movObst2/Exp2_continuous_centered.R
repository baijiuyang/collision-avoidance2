# ======================================================================
#  Exp 2 (Cohen_movObst2) -- passing-order GLMM
#  Predictors treated as CONTINUOUS, centered + normalized (z-scored).
#  Notes:
#   - z-scoring does NOT change model fit, Type II tests, predictions, or
#     switching points. It makes the interaction-model coefficients
#     interpretable (per 1 SD) and removes the ill-conditioning warnings.
#   - The linear-in-logit assumption is checked visually: observed cell
#     proportions (points) should lie on the fitted curve.
#   - Switching points and graph axes are reported in ORIGINAL units.
#   - Focal switching variable is PATH ANGLE.
# ======================================================================
library(dplyr)
library(lme4)
library(car)
library(ggplot2)

set.seed(88)
setwd("C:/Users/baiji/OneDrive/Academic/Dissertation/Codes/Bai_movObst2")

data <- read.csv("Cohen_movObst2_pass_order.csv")
data$pass_order[data$pass_order == -1] <- 0   # behind = 0, ahead = 1
data$obst_angle <- abs(data$obst_angle)        # path-angle magnitude

# ---- Center + normalize (mean 0, SD 1) ------------------------------
a_mu <- mean(data$obst_angle); a_sd <- sd(data$obst_angle)
d_mu <- mean(data$obst_dist);  d_sd <- sd(data$obst_dist)
data$angle_z <- (data$obst_angle - a_mu) / a_sd
data$dist_z  <- (data$obst_dist  - d_mu) / d_sd

# ---- Model ----------------------------------------------------------
m <- glmer(
  pass_order ~ angle_z * dist_z + (1 | subj_id),
  data = data, family = binomial(link = "logit")
)
summary(m)   # coefficients are per 1 SD of each predictor
Anova(m)     # Type-II Wald chi-square tests (invariant to scaling)

b <- fixef(m)

# ---- Switching points (solved in z-space, returned in ORIGINAL units)
# path angle at P(ahead) = 0.50, per initial distance  (focal)
sw_angle <- function(dist) {
  dz <- (dist - d_mu) / d_sd
  az <- -(b[["(Intercept)"]] + b[["dist_z"]] * dz) /
         (b[["angle_z"]] + b[["angle_z:dist_z"]] * dz)
  az * a_sd + a_mu
}
# initial distance at P(ahead) = 0.50, per path angle  (reverse)
sw_dist <- function(angle) {
  az <- (angle - a_mu) / a_sd
  dz <- -(b[["(Intercept)"]] + b[["angle_z"]] * az) /
         (b[["dist_z"]] + b[["angle_z:dist_z"]] * az)
  dz * d_sd + d_mu
}

angles <- sort(unique(data$obst_angle))
dists  <- sort(unique(data$obst_dist))
switch_angle_tbl <- data.frame(obst_dist  = dists,  switch_angle = sapply(dists,  sw_angle))
switch_dist_tbl  <- data.frame(obst_angle = angles, switch_dist  = sapply(angles, sw_dist))
cat("\nSwitching angle (P=0.50) by initial distance:\n"); print(switch_angle_tbl)
cat("\nSwitching distance (P=0.50) by path angle:\n");     print(switch_dist_tbl)

# ---- Logistic-function graphs (axes in ORIGINAL units) --------------
predict_fixed <- function(model, grid, form) {
  X   <- model.matrix(form, grid)
  eta <- as.vector(X %*% fixef(model))
  se  <- sqrt(diag(X %*% as.matrix(vcov(model)) %*% t(X)))
  grid$p     <- plogis(eta)
  grid$lower <- plogis(eta - 1.96 * se)
  grid$upper <- plogis(eta + 1.96 * se)
  grid
}
zc <- function(g) {
  g$angle_z <- (g$obst_angle - a_mu) / a_sd
  g$dist_z  <- (g$obst_dist  - d_mu) / d_sd
  g
}

obs <- data %>% group_by(obst_angle, obst_dist) %>%
  summarise(p = mean(pass_order), .groups = "drop")

## Graph A: x = path angle, one curve per initial distance (focal)
gridA <- zc(expand.grid(
  obst_angle = seq(min(data$obst_angle), max(data$obst_angle), length.out = 200),
  obst_dist  = dists))
gridA <- predict_fixed(m, gridA, ~ angle_z * dist_z)
swA <- subset(switch_angle_tbl,
              switch_angle >= min(data$obst_angle) & switch_angle <= max(data$obst_angle))

pA <- ggplot(gridA, aes(obst_angle, p, color = factor(obst_dist), fill = factor(obst_dist))) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(data = obs, aes(obst_angle, p, color = factor(obst_dist)),
             inherit.aes = FALSE, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_vline(data = swA, aes(xintercept = switch_angle, color = factor(obst_dist)),
             linetype = "dotted", show.legend = FALSE) +
  coord_cartesian(xlim = range(data$obst_angle)) +
  labs(x = "Path angle", y = "P(pass ahead)",
       color = "Initial distance", fill = "Initial distance",
       title = "Exp 2: logistic function over path angle") +
  theme_classic()
print(pA)

## Graph B: x = initial distance, one curve per path angle
gridB <- zc(expand.grid(
  obst_dist  = seq(min(data$obst_dist), max(data$obst_dist), length.out = 200),
  obst_angle = angles))
gridB <- predict_fixed(m, gridB, ~ angle_z * dist_z)
swB <- subset(switch_dist_tbl,
              switch_dist >= min(data$obst_dist) & switch_dist <= max(data$obst_dist))

pB <- ggplot(gridB, aes(obst_dist, p, color = factor(obst_angle), fill = factor(obst_angle))) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(data = obs, aes(obst_dist, p, color = factor(obst_angle)),
             inherit.aes = FALSE, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_vline(data = swB, aes(xintercept = switch_dist, color = factor(obst_angle)),
             linetype = "dotted", show.legend = FALSE) +
  coord_cartesian(xlim = range(data$obst_dist)) +
  labs(x = "Initial distance", y = "P(pass ahead)",
       color = "Path angle", fill = "Path angle",
       title = "Exp 2: logistic function over initial distance") +
  theme_classic()
print(pB)

ggsave("Exp2_logistic_angle.png", pA, width = 6, height = 4, dpi = 300)
ggsave("Exp2_logistic_dist.png",  pB, width = 6, height = 4, dpi = 300)
