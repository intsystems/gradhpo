========
Обзор
========

Постановка задачи
=================

Задача билевел-оптимизации в контексте машинного обучения формулируется
как вложенная задача оптимизации:

.. math::

   \min_{\lambda}\; L_{\mathrm{val}}\bigl(w^*(\lambda),\,\lambda\bigr),
   \qquad
   w^*(\lambda) = \arg\min_{w}\; L_{\mathrm{train}}(w,\,\lambda),

где :math:`w` --- параметры модели (внутренний уровень),
:math:`\lambda` --- гиперпараметры (внешний уровень),
:math:`L_{\mathrm{train}}` и :math:`L_{\mathrm{val}}` --- функции потерь
на обучающей и валидационной выборках соответственно.

На практике решение :math:`w^*(\lambda)` недоступно аналитически,
поэтому его аппроксимируют конечным числом шагов оптимизации.
Обозначим один шаг внутреннего оптимизатора как
:math:`\Phi(w, \lambda; D)`, тогда после :math:`T` шагов получаем
:math:`w_T`, который зависит от :math:`\lambda` через всю траекторию.

Гиперградиент
=============

Полный гиперградиент :math:`\mathrm{d}L_{\mathrm{val}} / \mathrm{d}\lambda`
раскладывается через chain rule:

.. math::

   \frac{\mathrm{d}L_{\mathrm{val}}}{\mathrm{d}\lambda}
   = \underbrace{\frac{\partial L_{\mathrm{val}}}{\partial \lambda}}_{g_{\mathrm{FO}}}
   + \sum_{t=1}^{T}
     \underbrace{\alpha_t \cdot
       \prod_{s=t+1}^{T} A_s}_{} \cdot B_t,

где :math:`\alpha_t = \nabla_{w_t} L_{\mathrm{val}}(w_t)`,
:math:`A_s = \partial \Phi / \partial w` и
:math:`B_t = \partial \Phi / \partial \lambda`.

Вычисление полной суммы требует обратного прохода через все :math:`T`
шагов, что дорого по памяти и времени.  Библиотека GradHpO реализует
несколько short-horizon аппроксимаций этой суммы.

Реализованные алгоритмы
=======================

T1-T2 с DARTS
--------------

Алгоритм T1-T2 (Luketina et al., 2016) разделяет шаг обновления
параметров и гиперпараметров.  В нашей реализации для вычисления
:math:`B_t` используется DARTS-аппроксимация (Liu et al., 2018)
на основе конечных разностей:

.. math::

   B_t \approx
   \frac{\Phi(w, \lambda + \varepsilon e_i) - \Phi(w, \lambda - \varepsilon e_i)}
   {2\varepsilon}.

Это позволяет избежать явного дифференцирования через оптимизатор.

Greedy
------

Обобщённый жадный подход (Agarwal et al., 2021) учитывает :math:`T` шагов
с экспоненциальным затуханием:

.. math::

   \hat{d}_\lambda =
   \nabla_\lambda L_{\mathrm{val}}(w_T)
   + \sum_{t=1}^{T} \gamma^{T-t}\,
     \nabla_{w_t} L_{\mathrm{val}}(w_t) \cdot B_t.

Параметр :math:`\gamma \in (0, 1]` контролирует вклад ранних шагов.

HyperDistill
-------------

Online-метод с дистилляцией гиперградиента (Lee et al., ICLR 2022)
аппроксимирует полный SO-терм через EMA-точку :math:`w^*_t`:

.. math::

   w^*_t = p_t \cdot w^*_{t-1} + (1 - p_t) \cdot w_{t-1},
   \qquad
   p_t = \frac{\gamma - \gamma^t}{1 - \gamma^t}.

Гиперградиент на шаге :math:`t`:

.. math::

   g_t = g_{\mathrm{FO}} + \theta \cdot \frac{1 - \gamma^t}{1 - \gamma}
   \cdot v_t,

где :math:`v_t = \alpha_t \cdot \partial\Phi(w^*_t, \lambda) / \partial\lambda`,
а скаляр :math:`\theta` оценивается периодически через DrMAD-backward
(Algorithm 4 из статьи).

Бейзлайны
----------

- **FO (First-Order)**: использует только прямой градиент
  :math:`g_{\mathrm{FO}} = \partial L_{\mathrm{val}} / \partial \lambda`.
  Если :math:`\lambda` не входит в :math:`L_{\mathrm{val}}` напрямую,
  обновление нулевое.

- **One-Step**: учитывает :math:`B_t` только на последнем шаге,
  :math:`g = g_{\mathrm{FO}} + \alpha_T \cdot B_T`.
  Эквивалентен HyperDistill с :math:`\gamma = 0`.

Архитектура библиотеки
======================

Все алгоритмы наследуют от ``BilevelOptimizer`` и реализуют три метода:

.. code-block:: python

   class BilevelOptimizer(ABC):
       def init(self, params, hyperparams) -> BilevelState: ...
       def step(self, state, train_batch, val_batch,
                train_loss_fn, val_loss_fn) -> BilevelState: ...
       def compute_hypergradient(self, state, train_batch, val_batch,
                                 train_loss_fn, val_loss_fn) -> PyTree: ...

Состояние оптимизации хранится в ``BilevelState`` --- неизменяемом
контейнере с полями ``params``, ``hyperparams``, ``inner_opt_state``,
``outer_opt_state``, ``step`` и ``metadata``.
