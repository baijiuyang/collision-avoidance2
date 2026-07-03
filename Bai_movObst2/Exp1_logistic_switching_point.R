# ======================================================================
#  Exp 1  (Cohen_movObst1) -- passing-order GLMM
#  Deliverables for Bill:
#    (1) logistic-regression output
#    (2) a logistic-function graph for EACH predictor
#    (3) the "switching point": predictor value where P(ahead) = 0.50
# ======================================================================
library(dplyr)
library(lme4)
library(car)        # Anova
library(ggplot2)

set.seed(88)
setwd("C:/Users/baiji/OneDrive/Academic/Dissertation/Codes/Bai_movObst2")

data <- read.csv("Cohen_movObst1_pass_order.csv")

# Outcome: behind = 0, ahead = 1
data$pass_order[data$pass_order == -1] <- 0
# Use the magnitude of the path angle
data$obst_angle <- abs(data$obst_angle)

# ----------------------------------------------------------------------
# SECTION 1 -- categorical model (what you already ran) -> ANOVA
# Factors are correct for the "does this predictor matter" test, so keep
# this for the manuscript's hypothesis test.
# ----------------------------------------------------------------------
data_cat <- data
data_cat$obst_angle <- factor(data_cat$obst_angle)
data_cat$obst_speed <- factor(data_cat$obst_speed)

m_cat <- glmer(
  pass_order ~ obst_angle * obst_speed + (1 | subj_id),
  data = data_cat, family = binomial(link = "logit")
)
Anova(m_cat)        # Type-II Wald chi-square tests

# ----------------------------------------------------------------------
# SECTION 2 -- continuous model -> logistic curve + switching point
# IMPORTANT: a 50% crossing and a smooth logistic curve only exist when
# the predictor is numeric, so here obst_angle / obst_speed stay numeric.
# ----------------------------------------------------------------------
data$obst_angle <- as.numeric(data$obst_angle)
data$obst_speed <- as.numeric(data$obst_speed)

m_cont <- glmer(
  pass_order ~ obst_angle * obst_speed + (1 | subj_id),
  data = data, family = binomial(link = "logit")
)

# (1) Output ----------------------------------------------------------
summary(m_cont)     # fixed-effect coefficients (log-odds) = the logistic fit
Anova(m_cont)       # Type-II Wald tests on the continuous predictors

# ----------------------------------------------------------------------
# SECTION 2b -- rescaled refit (clears the "rescale variables" warning)
# z-score the predictors so the fixed effects are well-conditioned.
# Predictions / switching points are unchanged; this just gives a clean
# coefficient table and standard errors for the manuscript.
# ----------------------------------------------------------------------
data$angle_z <- as.numeric(scale(data$obst_angle))
data$speed_z <- as.numeric(scale(data$obst_speed))

m_z <- glmer(
  pass_order ~ angle_z * speed_z + (1 | subj_id),
  data = data, family = binomial(link = "logit")
)
summary(m_z)   # clean coefficients (per 1 SD of each predictor)
Anova(m_z)

# Switching speed in ORIGINAL units, confirmed from the rescaled model:
bz <- fixef(m_z)
a_mu <- mean(data$obst_angle); a_sd <- sd(data$obst_angle)
s_mu <- mean(data$obst_speed); s_sd <- sd(data$obst_speed)
sw_speed_z <- function(angle) {
  az <- (angle - a_mu) / a_sd
  sz <- -(bz[["(Intercept)"]] + bz[["angle_z"]] * az) /
         (bz[["speed_z"]] + bz[["angle_z:speed_z"]] * az)
  sz * s_sd + s_mu
}
cat("\nSwitching speed (rescaled model, original units):\n")
print(data.frame(obst_angle = angles, switch_speed = sapply(angles, sw_speed_z)))

# (3) Switching point: obstacle speed at P(ahead) = 0.50 --------------
# logit(p) = b0 + b1*angle + b2*speed + b3*angle*speed
# set = 0  ->  speed* = -(b0 + b1*angle) / (b2 + b3*angle)
b <- fixef(m_cont)
sw_speed <- function(angle)
  -(b[["(Intercept)"]] + b[["obst_angle"]] * angle) /
   (b[["obst_speed"]]  + b[["obst_angle:obst_speed"]] * angle)

angles <- sort(unique(data$obst_angle))
switch_points <- data.frame(obst_angle   = angles,
                            switch_speed = sapply(angles, sw_speed))
cat("\nSwitching speed (P=0.50) at each path angle:\n")
print(switch_points)

# ... and the reverse: PATH ANGLE at P(ahead) = 0.50, per speed level
# set logit = 0  ->  angle* = -(b0 + b2*speed) / (b1 + b3*speed)
sw_angle <- function(speed)
  -(b[["(Intercept)"]] + b[["obst_speed"]] * speed) /
   (b[["obst_angle"]]  + b[["obst_angle:obst_speed"]] * speed)

speeds_all <- sort(unique(data$obst_speed))
switch_points_angle <- data.frame(obst_speed   = speeds_all,
                                  switch_angle = sapply(speeds_all, sw_angle))
cat("\nSwitching path angle (P=0.50) at each obstacle speed:\n")
print(switch_points_angle)

# Optional: one overall switching speed (ignore angle) ----------------
# m_one <- glmer(pass_order ~ obst_speed + (1|subj_id), data, binomial)
# -fixef(m_one)[1] / fixef(m_one)[2]

# (2) Logistic-function graphs ----------------------------------------
# Population-level (fixed-effect) predictions + 95% CI:
predict_fixed <- function(model, grid, form) {
  X   <- model.matrix(form, grid)
  eta <- as.vector(X %*% fixef(model))
  se  <- sqrt(diag(X %*% as.matrix(vcov(model)) %*% t(X)))
  grid$p     <- plogis(eta)
  grid$lower <- plogis(eta - 1.96 * se)
  grid$upper <- plogis(eta + 1.96 * se)
  grid
}

# observed proportion per design cell (overlaid as points)
obs <- data %>%
  group_by(obst_angle, obst_speed) %>%
  summarise(p = mean(pass_order), n = n(), .groups = "drop")

## Graph A: x = obstacle speed, one curve per path angle (the focal one)
gridA <- expand.grid(
  obst_speed = seq(min(data$obst_speed), max(data$obst_speed), length.out = 200),
  obst_angle = angles
)
gridA <- predict_fixed(m_cont, gridA, ~ obst_angle * obst_speed)

pA <- ggplot(gridA, aes(obst_speed, p, color = factor(obst_angle),
                        fill = factor(obst_angle))) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(data = obs, aes(obst_speed, p, color = factor(obst_angle)),
             inherit.aes = FALSE, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_vline(data = switch_points,
             aes(xintercept = switch_speed, color = factor(obst_angle)),
             linetype = "dotted", show.legend = FALSE) +
  labs(x = "Obstacle speed", y = "P(pass ahead)",
       color = "Path angle", fill = "Path angle",
       title = "Exp 1: logistic function over obstacle speed") +
  theme_classic()
print(pA)

## Graph B: x = path angle, one curve per obstacle speed
speeds <- sort(unique(data$obst_speed))
gridB <- expand.grid(
  obst_angle = seq(min(data$obst_angle), max(data$obst_angle), length.out = 200),
  obst_speed = speeds
)
gridB <- predict_fixed(m_cont, gridB, ~ obst_angle * obst_speed)

# only draw switching lines that fall within the tested angle range
sw_angle_in <- subset(switch_points_angle,
                      switch_angle >= min(data$obst_angle) &
                      switch_angle <= max(data$obst_angle))

pB <- ggplot(gridB, aes(obst_angle, p, color = factor(obst_speed),
                        fill = factor(obst_speed))) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(data = obs, aes(obst_angle, p, color = factor(obst_speed)),
             inherit.aes = FALSE, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_vline(data = sw_angle_in,
             aes(xintercept = switch_angle, color = factor(obst_speed)),
             linetype = "dotted", show.legend = FALSE) +
  coord_cartesian(xlim = range(data$obst_angle)) +
  labs(x = "Path angle", y = "P(pass ahead)",
       color = "Obstacle speed", fill = "Obstacle speed",
       title = "Exp 1: logistic function over path angle") +
  theme_classic()
print(pB)

# Save figures (optional)
ggsave("Exp1_logistic_speed.png", pA, width = 6, height = 4, dpi = 300)
ggsave("Exp1_logistic_angle.png", pB, width = 6, height = 4, dpi = 300)
