# ======================================================================
#  Exp 1 (Cohen_movObst1) -- passing-order GLMM
#  Predictors treated as CONTINUOUS, centered + normalized (z-scored).
#  Notes:
#   - z-scoring does NOT change model fit, Type II tests, predictions, or
#     switching points. It makes the interaction-model coefficients
#     interpretable (per 1 SD) and removes the ill-conditioning warnings.
#   - The linear-in-logit assumption is checked visually: observed cell
#     proportions (points) should lie on the fitted curve.
#   - Switching points and graph axes are reported in ORIGINAL units.
# ======================================================================
library(dplyr)
library(lme4)
library(car)
library(ggplot2)

set.seed(88)
setwd("C:/Users/baiji/OneDrive/Academic/Dissertation/Codes/Bai_movObst2")

data <- read.csv("Cohen_movObst1_pass_order.csv")
data$pass_order[data$pass_order == -1] <- 0   # behind = 0, ahead = 1
data$obst_angle <- abs(data$obst_angle)        # path-angle magnitude

# ---- Center + normalize (mean 0, SD 1) ------------------------------
a_mu <- mean(data$obst_angle); a_sd <- sd(data$obst_angle)
s_mu <- mean(data$obst_speed); s_sd <- sd(data$obst_speed)
data$angle_z <- (data$obst_angle - a_mu) / a_sd
data$speed_z <- (data$obst_speed - s_mu) / s_sd

# ---- Model ----------------------------------------------------------
m <- glmer(
  pass_order ~ angle_z * speed_z + (1 | subj_id),
  data = data, family = binomial(link = "logit")
)
summary(m)   # coefficients are per 1 SD of each predictor
Anova(m)     # Type-II Wald chi-square tests (invariant to scaling)

b <- fixef(m)

# ---- Switching points (solved in z-space, returned in ORIGINAL units)
# obstacle speed at P(ahead) = 0.50, per path angle
sw_speed <- function(angle) {
  az <- (angle - a_mu) / a_sd
  sz <- -(b[["(Intercept)"]] + b[["angle_z"]] * az) /
         (b[["speed_z"]] + b[["angle_z:speed_z"]] * az)
  sz * s_sd + s_mu
}
# path angle at P(ahead) = 0.50, per obstacle speed
sw_angle <- function(speed) {
  sz <- (speed - s_mu) / s_sd
  az <- -(b[["(Intercept)"]] + b[["speed_z"]] * sz) /
         (b[["angle_z"]] + b[["angle_z:speed_z"]] * sz)
  az * a_sd + a_mu
}

angles <- sort(unique(data$obst_angle))
speeds <- sort(unique(data$obst_speed))
switch_speed_tbl <- data.frame(obst_angle = angles, switch_speed = sapply(angles, sw_speed))
switch_angle_tbl <- data.frame(obst_speed = speeds, switch_angle = sapply(speeds, sw_angle))
cat("\nSwitching speed (P=0.50) by path angle:\n");   print(switch_speed_tbl)
cat("\nSwitching angle (P=0.50) by obstacle speed:\n"); print(switch_angle_tbl)

# ---- Logistic-function graphs (axes in ORIGINAL units) --------------
# population-level (fixed-effect) prediction + 95% CI band
predict_fixed <- function(model, grid, form) {
  X   <- model.matrix(form, grid)
  eta <- as.vector(X %*% fixef(model))
  se  <- sqrt(diag(X %*% as.matrix(vcov(model)) %*% t(X)))
  grid$p     <- plogis(eta)
  grid$lower <- plogis(eta - 1.96 * se)
  grid$upper <- plogis(eta + 1.96 * se)
  grid
}
# add z-scored columns to a grid built in original units
zc <- function(g) {
  g$angle_z <- (g$obst_angle - a_mu) / a_sd
  g$speed_z <- (g$obst_speed - s_mu) / s_sd
  g
}

# observed proportion per design cell (visual check on linear-in-logit)
obs <- data %>% group_by(obst_angle, obst_speed) %>%
  summarise(p = mean(pass_order), .groups = "drop")

## Graph A: x = obstacle speed, one curve per path angle
gridA <- zc(expand.grid(
  obst_speed = seq(min(data$obst_speed), max(data$obst_speed), length.out = 200),
  obst_angle = angles))
gridA <- predict_fixed(m, gridA, ~ angle_z * speed_z)
swA <- subset(switch_speed_tbl,
              switch_speed >= min(data$obst_speed) & switch_speed <= max(data$obst_speed))

pA <- ggplot(gridA, aes(obst_speed, p, color = factor(obst_angle), fill = factor(obst_angle))) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(data = obs, aes(obst_speed, p, color = factor(obst_angle)),
             inherit.aes = FALSE, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_vline(data = swA, aes(xintercept = switch_speed, color = factor(obst_angle)),
             linetype = "dotted", show.legend = FALSE) +
  coord_cartesian(xlim = range(data$obst_speed)) +
  labs(x = "Obstacle speed", y = "P(pass ahead)",
       color = "Path angle", fill = "Path angle",
       title = "Exp 1: logistic function over obstacle speed") +
  theme_classic()
print(pA)

## Graph B: x = path angle, one curve per obstacle speed
gridB <- zc(expand.grid(
  obst_angle = seq(min(data$obst_angle), max(data$obst_angle), length.out = 200),
  obst_speed = speeds))
gridB <- predict_fixed(m, gridB, ~ angle_z * speed_z)
swB <- subset(switch_angle_tbl,
              switch_angle >= min(data$obst_angle) & switch_angle <= max(data$obst_angle))

pB <- ggplot(gridB, aes(obst_angle, p, color = factor(obst_speed), fill = factor(obst_speed))) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(data = obs, aes(obst_angle, p, color = factor(obst_speed)),
             inherit.aes = FALSE, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_vline(data = swB, aes(xintercept = switch_angle, color = factor(obst_speed)),
             linetype = "dotted", show.legend = FALSE) +
  coord_cartesian(xlim = range(data$obst_angle)) +
  labs(x = "Path angle", y = "P(pass ahead)",
       color = "Obstacle speed", fill = "Obstacle speed",
       title = "Exp 1: logistic function over path angle") +
  theme_classic()
print(pB)

ggsave("Exp1_logistic_speed.png", pA, width = 6, height = 4, dpi = 300)
ggsave("Exp1_logistic_angle.png", pB, width = 6, height = 4, dpi = 300)
