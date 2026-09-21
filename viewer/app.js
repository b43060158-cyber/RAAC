const DEFAULT_PATH = "";
const urlParams = new URLSearchParams(window.location.search);
const initialPath = urlParams.get("path") || localStorage.getItem("masViewer.runPath") || DEFAULT_PATH;
const initialDebugMode =
  urlParams.get("debug") === "1" || urlParams.get("debug") === "true"
    ? true
    : localStorage.getItem("masViewer.debugMode") === "true";

const state = {
  runPath: initialPath,
  availableRuns: [],
  runsLoading: false,
  run: null,
  questionDetail: null,
  selectedQuestionKey: "",
  answerRound: null,
  reviewRound: null,
  selectedReviewId: "",
  debugMode: initialDebugMode,
  search: "",
  searchDraft: "",
  correctnessFilter: "all",
  changeFilter: "all",
  loading: false,
  detailLoading: false,
  questionListScrollTop: 0,
  pageScrollTop: 0,
  scrollRestore: null,
  error: "",
  status: "输入运行结果目录路径后加载。",
};

const app = document.querySelector("#app");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function encodeQuery(value) {
  return encodeURIComponent(value ?? "");
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${Math.round(value * 1000) / 10}%`;
}

function formatConfidence(value) {
  if (value === null || value === undefined || value === "") return "NAN";
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return escapeHtml(value);
  return numeric.toFixed(2).replace(/\.?0+$/, "");
}

function correctnessBadge(question) {
  if (question.excluded_from_accuracy) return `<span class="badge warn">排除</span>`;
  if (question.is_tie) return `<span class="badge warn">平局</span>`;
  if (question.is_correct === true) return `<span class="badge good">正确</span>`;
  if (question.is_correct === false) return `<span class="badge bad">错误</span>`;
  return `<span class="badge">未评估</span>`;
}

function optionMap(question) {
  const map = new Map();
  for (const option of question?.options || []) {
    map.set(String(option.option_id), String(option.text ?? ""));
  }
  return map;
}

function answerText(answer, question) {
  const selected = Array.isArray(answer?.selected_option_ids) ? answer.selected_option_ids : [];
  const options = optionMap(question);
  if (selected.length) {
    return selected
      .map((id) => {
        const key = String(id);
        const text = options.get(key);
        return text ? `${key}. ${text}` : key;
      })
      .join("; ");
  }
  if (answer?.final_answer) return String(answer.final_answer);
  if (answer?.code) {
    const firstLine = String(answer.code).split("\n")[0] || "代码答案";
    return firstLine.length > 100 ? `${firstLine.slice(0, 100)}...` : firstLine;
  }
  return "未提供答案";
}

function filteredQuestions() {
  const questions = state.run?.questions || [];
  const search = state.search.trim().toLowerCase();
  return questions.filter((question) => {
    const haystack = [
      question.question_key,
      question.question_id,
      question.dataset_name,
      question.task_type,
      question.question_preview,
      question.selected_agent_id,
      question.selected_answer,
      question.selected_confidence,
    ]
      .join(" ")
      .toLowerCase();
    if (search && !haystack.includes(search)) return false;
    if (state.correctnessFilter === "correct" && question.is_correct !== true) return false;
    if (state.correctnessFilter === "wrong" && question.is_correct !== false) return false;
    if (state.correctnessFilter === "tie" && !question.is_tie) return false;
    if (state.correctnessFilter === "excluded" && !question.excluded_from_accuracy) return false;
    if (state.changeFilter === "changed" && !question.has_changes) return false;
    if (state.changeFilter === "unchanged" && question.has_changes) return false;
    return true;
  });
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `请求失败：${response.status}`);
  }
  return payload;
}

function runOptionLabel(run) {
  const countLabel =
    run.evaluated_question_count === run.question_count
      ? `${run.question_count}题`
      : `${run.evaluated_question_count}/${run.question_count}题已评估`;
  const accuracyLabel = typeof run.accuracy === "number" ? ` · ${formatPercent(run.accuracy)}` : "";
  return `${run.name} · ${countLabel}${accuracyLabel}`;
}

async function loadAvailableRuns() {
  state.runsLoading = true;
  render();
  try {
    const payload = await fetchJson("/api/runs");
    state.availableRuns = payload.runs || [];
    state.runsLoading = false;
    if (!urlParams.has("path") && !localStorage.getItem("masViewer.runPath") && state.availableRuns[0]?.path) {
      state.runPath = state.availableRuns[0].path;
    }
    render();
  } catch (error) {
    state.runsLoading = false;
    state.status = `运行列表加载失败：${error.message}`;
    render();
  }
}

async function loadRun(path) {
  state.loading = true;
  state.error = "";
  state.status = "正在读取运行结果...";
  state.questionDetail = null;
  state.search = "";
  state.searchDraft = "";
  render();
  try {
    const payload = await fetchJson(`/api/run?path=${encodeQuery(path)}`);
    state.run = payload;
    state.runPath = payload.run_path;
    localStorage.setItem("masViewer.runPath", state.runPath);
    const firstQuestion = payload.questions?.[0]?.question_key || "";
    state.selectedQuestionKey = firstQuestion;
    state.status = `已加载 ${payload.run_id}，共 ${payload.question_count} 道题。`;
    state.loading = false;
    render();
    if (firstQuestion) {
      await loadQuestion(firstQuestion);
    }
  } catch (error) {
    state.loading = false;
    state.error = error.message;
    state.status = "加载失败。";
    render();
  }
}

async function loadQuestion(questionKey) {
  if (!state.run) return;
  state.detailLoading = true;
  state.error = "";
  state.selectedQuestionKey = questionKey;
  state.selectedReviewId = "";
  syncQuestionSelection();
  try {
    const payload = await fetchJson(
      `/api/question?path=${encodeQuery(state.run.run_path)}&question_key=${encodeQuery(questionKey)}`,
    );
    state.questionDetail = payload;
    const rounds = payload.question?.rounds || [];
    const firstRound = rounds[0]?.round_index ?? null;
    state.answerRound = firstRound;
    state.reviewRound = firstRound;
    state.detailLoading = false;
    render();
    setTimeout(() => {
      state.scrollRestore = null;
    }, 220);
  } catch (error) {
    state.detailLoading = false;
    state.error = error.message;
    render();
    setTimeout(() => {
      state.scrollRestore = null;
    }, 220);
  }
}

function renderSummary() {
  const summary = state.run?.summary || {};
  return `
    <section class="panel">
      <h2 class="panel-title">运行概览</h2>
      <div class="summary-grid">
        <div class="metric"><div class="metric-label">题目数</div><div class="metric-value">${escapeHtml(state.run?.question_count ?? "-")}</div></div>
        <div class="metric"><div class="metric-label">准确率</div><div class="metric-value">${escapeHtml(formatPercent(summary.accuracy))}</div></div>
        <div class="metric"><div class="metric-label">正确题</div><div class="metric-value">${escapeHtml(summary.correct_questions ?? "-")}</div></div>
        <div class="metric"><div class="metric-label">轮次</div><div class="metric-value">${escapeHtml((state.run?.round_indexes || []).join(", ") || "-")}</div></div>
      </div>
      <div class="badge-row">
        ${(state.run?.agents || []).map((agent) => `<span class="badge">${escapeHtml(agent)}</span>`).join("")}
      </div>
    </section>
  `;
}

function debugPanel(title, value) {
  const text = String(value || "").trim();
  if (!state.debugMode || !text) return "";
  return `
    <details class="debug-box" open>
      <summary>${escapeHtml(title)}</summary>
      <div class="debug-content">${escapeHtml(text)}</div>
    </details>
  `;
}

function renderFilters() {
  return `
    <section class="panel filters">
      <form id="searchForm">
        <input class="search-input" id="searchInput" value="${escapeHtml(state.searchDraft)}" placeholder="搜索题号、题干、agent、答案" />
      </form>
      <div class="filter-row">
        <select class="select" id="correctnessFilter">
          <option value="all" ${state.correctnessFilter === "all" ? "selected" : ""}>全部结果</option>
          <option value="correct" ${state.correctnessFilter === "correct" ? "selected" : ""}>只看正确</option>
          <option value="wrong" ${state.correctnessFilter === "wrong" ? "selected" : ""}>只看错误</option>
          <option value="tie" ${state.correctnessFilter === "tie" ? "selected" : ""}>只看平局</option>
          <option value="excluded" ${state.correctnessFilter === "excluded" ? "selected" : ""}>只看排除</option>
        </select>
        <select class="select" id="changeFilter">
          <option value="all" ${state.changeFilter === "all" ? "selected" : ""}>全部变化</option>
          <option value="changed" ${state.changeFilter === "changed" ? "selected" : ""}>发生改答</option>
          <option value="unchanged" ${state.changeFilter === "unchanged" ? "selected" : ""}>没有改答</option>
        </select>
      </div>
    </section>
  `;
}

function renderQuestionList() {
  const questions = filteredQuestions();
  if (!state.run) {
    return `<section class="empty-state">尚未加载运行结果。</section>`;
  }
  if (!questions.length) {
    return `<section class="empty-state">没有符合筛选条件的题目。</section>`;
  }
  return `
    <section class="question-list" aria-label="题目列表">
      ${questions
        .map(
          (question) => `
            <div
              class="question-item ${question.question_key === state.selectedQuestionKey ? "is-active" : ""}"
              data-question-key="${escapeHtml(question.question_key)}"
              role="button"
              tabindex="0"
            >
              <div class="question-line">
                <div class="question-key">${escapeHtml(question.question_key)}</div>
                ${correctnessBadge(question)}
              </div>
              <div class="question-preview">${escapeHtml(question.question_preview || "无题干预览")}</div>
              <div class="badge-row">
                <span class="badge">${escapeHtml(question.task_type)}</span>
                <span class="badge">选中 ${escapeHtml(question.selected_agent_id || "-")}</span>
                <span class="badge confidence">置信度 ${formatConfidence(question.selected_confidence)}</span>
                ${
                  question.has_changes
                    ? `<span class="badge warn">改答 ${escapeHtml(question.changed_agents.join(", "))}</span>`
                    : `<span class="badge">未改答</span>`
                }
              </div>
            </div>
          `,
        )
        .join("")}
    </section>
  `;
}

function renderOptions(question) {
  if (!Array.isArray(question.options) || !question.options.length) return "";
  const correct = new Set(question.correct_option_ids || []);
  return `
    <div class="option-list">
      ${question.options
        .map(
          (option) => `
            <div class="option">
              <div class="option-id">${escapeHtml(option.option_id)}${correct.has(option.option_id) ? " ✓" : ""}</div>
              <div>${escapeHtml(option.text)}</div>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderQuestionHero(detail) {
  const question = detail.question;
  const evaluation = detail.evaluation || {};
  const title = question.question || question.prompt || question.question_key;
  return `
    <section class="hero-panel">
      <div class="question-heading">
        <div>
          <h1 class="question-title">${escapeHtml(question.question_key)}</h1>
          <div class="question-meta">
            <span class="badge">${escapeHtml(question.dataset_name)}</span>
            <span class="badge">${escapeHtml(question.task_type)}</span>
            ${correctnessBadge({
              is_correct: evaluation.is_correct,
              is_tie: evaluation.is_tie,
              excluded_from_accuracy: evaluation.excluded_from_accuracy,
            })}
            <span class="badge">最终 ${escapeHtml(evaluation.selected_agent_id || "-")}</span>
          </div>
        </div>
        <div class="score-line">
          预测：${escapeHtml((evaluation.predicted_option_ids || []).join(", ") || evaluation.predicted_final_answer || "-")}<br />
          正确：${escapeHtml((evaluation.correct_option_ids || question.correct_option_ids || []).join(", ") || (question.acceptable_answers || []).join(", ") || "-")}<br />
          置信度：${formatConfidence(
            (question.rounds?.at(-1)?.agent_results || []).find((result) => result.agent_id === evaluation.selected_agent_id)
              ?.answer?.confidence,
          )}
        </div>
      </div>
      <div class="question-body">${escapeHtml(title)}</div>
      ${question.entry_point ? `<div class="badge-row"><span class="badge">entry_point: ${escapeHtml(question.entry_point)}</span></div>` : ""}
      ${renderOptions(question)}
      ${question.test ? `<pre class="code-block">${escapeHtml(question.test)}</pre>` : ""}
    </section>
  `;
}

function roundTabs(rounds, activeRound, type) {
  return `
    <div class="tabs" role="tablist">
      ${rounds
        .map(
          (round) => `
            <button class="tab-button ${round.round_index === activeRound ? "is-active" : ""}" data-round-type="${type}" data-round-index="${escapeHtml(round.round_index)}">
              第 ${escapeHtml(round.round_index)} 轮
            </button>
          `,
        )
        .join("")}
    </div>
  `;
}

function findRound(question, roundIndex) {
  return (question.rounds || []).find((round) => String(round.round_index) === String(roundIndex));
}

function renderAnswerPhase(detail) {
  const question = detail.question;
  const rounds = question.rounds || [];
  const activeRound = state.answerRound ?? rounds[0]?.round_index;
  const round = findRound(question, activeRound) || rounds[0];
  if (!round) return "";
  return `
    <section class="phase-section">
      <div class="section-head">
        <h2 class="section-title">答题阶段</h2>
        ${roundTabs(rounds, activeRound, "answer")}
      </div>
      <div class="agent-grid">
        ${(round.agent_results || [])
          .map((result) => {
            const answer = result.answer || {};
            return `
              <article class="agent-card">
                <div class="agent-head">
                  <div>
                    <div class="agent-name">${escapeHtml(result.agent_id)}</div>
                    <div class="score-line">总分 ${escapeHtml(result.total_score ?? "-")} / 均分 ${escapeHtml(Number(result.average_score ?? 0).toFixed(2))} / 置信度 ${formatConfidence(answer.confidence)}</div>
                  </div>
                  <div class="badge-row">
                    ${result["advers-agent"] ? `<span class="badge bad">adversarial</span>` : ""}
                    ${answer.changed_answer ? `<span class="badge warn">改答</span>` : `<span class="badge">未改答</span>`}
                  </div>
                </div>
                <div class="answer-value">${escapeHtml(answerText(answer, question))}</div>
                <div class="reasoning">${escapeHtml(answer.reasoning || "无理由")}</div>
                ${debugPanel("Debug: chain_of_thought", answer.chain_of_thought)}
                ${
                  answer.code
                    ? `<pre class="code-block">${escapeHtml(answer.code)}</pre>`
                    : ""
                }
                ${
                  answer.changed_answer || answer.change_summary
                    ? `<div class="change-box">
                        <strong>改答说明</strong><br />
                        驱动 agent：${escapeHtml((answer.change_drivers || []).join(", ") || "-")}<br />
                        ${escapeHtml(answer.change_summary || "-")}
                      </div>`
                    : ""
                }
              </article>
            `;
          })
          .join("")}
      </div>
    </section>
  `;
}

function collectReviews(round) {
  const reviews = [];
  for (const result of round?.agent_results || []) {
    for (const review of result.reviews_given || []) {
      reviews.push(review);
    }
  }
  return reviews;
}

function reviewKey(review) {
  return `${review.reviewer_agent_id}->${review.target_agent_id}`;
}

function renderReviewPhase(detail) {
  const question = detail.question;
  const rounds = question.rounds || [];
  const activeRound = state.reviewRound ?? rounds[0]?.round_index;
  const round = findRound(question, activeRound) || rounds[0];
  if (!round) return "";
  const agents = (round.agent_results || []).map((result) => result.agent_id);
  const reviews = collectReviews(round);
  const reviewByPair = new Map(reviews.map((review, index) => [reviewKey(review), { ...review, index }]));
  const selectedReview =
    reviews.find((review, index) => String(index) === String(state.selectedReviewId)) || reviews[0];
  return `
    <section class="phase-section">
      <div class="section-head">
        <h2 class="section-title">评审阶段</h2>
        ${roundTabs(rounds, activeRound, "review")}
      </div>
      <div class="review-layout">
        <div class="matrix-wrap">
          <table class="review-matrix">
            <thead>
              <tr>
                <th>Reviewer \\ Target</th>
                ${agents.map((agent) => `<th>${escapeHtml(agent)}</th>`).join("")}
              </tr>
            </thead>
            <tbody>
              ${agents
                .map(
                  (reviewer) => `
                    <tr>
                      <th>${escapeHtml(reviewer)}</th>
                      ${agents
                        .map((target) => {
                          if (reviewer === target) return `<td class="cell-empty">-</td>`;
                          const review = reviewByPair.get(`${reviewer}->${target}`);
                          if (!review) return `<td class="cell-empty">无</td>`;
                          return `
                            <td>
                              <button class="cell-button ${escapeHtml(review.stance)}" data-review-id="${escapeHtml(review.index)}">
                                <span class="cell-score">${escapeHtml(review.score)}</span>
                                <span class="cell-stance">${escapeHtml(review.stance)}</span>
                              </button>
                            </td>
                          `;
                        })
                        .join("")}
                    </tr>
                  `,
                )
                .join("")}
            </tbody>
          </table>
        </div>
        <aside class="review-detail">
          ${
            selectedReview
              ? `
                <h3 class="review-title">${escapeHtml(selectedReview.reviewer_agent_id)} → ${escapeHtml(selectedReview.target_agent_id)}</h3>
                <div class="badge-row">
                  <span class="badge ${escapeHtml(selectedReview.stance)}">分数 ${escapeHtml(selectedReview.score)}</span>
                  <span class="badge ${escapeHtml(selectedReview.stance)}">${escapeHtml(selectedReview.stance)}</span>
                </div>
                <div class="review-reason">${escapeHtml(selectedReview.main_reason || "无评分理由")}</div>
                ${debugPanel("Debug: review chain_of_thought", selectedReview.chain_of_thought)}
              `
              : `<div class="empty-state">本轮没有评审记录。</div>`
          }
        </aside>
      </div>
    </section>
  `;
}

function renderDetail() {
  if (state.detailLoading && !state.questionDetail) {
    return `<section class="empty-state loading">正在读取题目详情...</section>`;
  }
  if (!state.run) {
    return `<section class="empty-state">加载 run 后会在这里展示每道题的答题阶段和评审阶段。</section>`;
  }
  if (!state.questionDetail) {
    return `<section class="empty-state">请选择一道题。</section>`;
  }
  return `
    ${state.detailLoading ? `<section class="panel loading-inline">正在读取题目详情...</section>` : ""}
    ${renderQuestionHero(state.questionDetail)}
    ${renderAnswerPhase(state.questionDetail)}
    ${renderReviewPhase(state.questionDetail)}
  `;
}

function syncQuestionSelection() {
  for (const item of app.querySelectorAll("[data-question-key]")) {
    item.classList.toggle("is-active", item.dataset.questionKey === state.selectedQuestionKey);
  }
}

function render() {
  const previousQuestionList = app.querySelector(".question-list");
  if (previousQuestionList && !state.scrollRestore) {
    state.questionListScrollTop = previousQuestionList.scrollTop;
  }
  if (!state.scrollRestore) {
    state.pageScrollTop = window.scrollY;
  }
  const nextQuestionListScrollTop = state.scrollRestore?.questionListScrollTop ?? state.questionListScrollTop;
  const nextPageScrollTop = state.scrollRestore?.pageScrollTop ?? state.pageScrollTop;

  app.innerHTML = `
    <main class="app-shell">
      <header class="topbar">
        <div class="topbar-inner">
          <form class="path-form" id="pathForm">
            <input class="path-input" id="pathInput" value="${escapeHtml(state.runPath)}" list="runOptions" aria-label="run目录路径" placeholder="${state.runsLoading ? "正在扫描 runs/..." : "选择 runs 下的结果或输入绝对路径"}" />
            <datalist id="runOptions">
              ${state.availableRuns
                .map(
                  (run) => `
                    <option value="${escapeHtml(run.path)}" label="${escapeHtml(runOptionLabel(run))}"></option>
                  `,
                )
                .join("")}
            </datalist>
            <button class="primary-button" type="submit" ${state.loading ? "disabled" : ""}>
              ${state.loading ? "加载中" : "加载结果"}
            </button>
            <button class="ghost-button debug-toggle ${state.debugMode ? "is-active" : ""}" id="debugToggle" type="button" aria-pressed="${state.debugMode ? "true" : "false"}">
              Debug ${state.debugMode ? "开" : "关"}
            </button>
          </form>
          <div class="status-line">${escapeHtml(state.status)}</div>
        </div>
      </header>
      <div class="workspace">
        <aside class="sidebar">
          ${state.run ? renderSummary() : `<section class="panel"><h2 class="panel-title">MAS 运行结果</h2><div class="question-preview">读取指定 run 目录，逐题查看多轮答题与互评。</div></section>`}
          ${renderFilters()}
          ${renderQuestionList()}
        </aside>
        <section class="detail">
          ${state.error ? `<div class="error-state">${escapeHtml(state.error)}</div>` : ""}
          ${renderDetail()}
        </section>
      </div>
    </main>
  `;

  const nextQuestionList = app.querySelector(".question-list");
  if (nextQuestionList) {
    nextQuestionList.scrollTop = nextQuestionListScrollTop;
  }
  window.scrollTo({ top: nextPageScrollTop, behavior: "auto" });

  const restoreScroll = () => {
    const latestQuestionList = app.querySelector(".question-list");
    if (latestQuestionList) {
      latestQuestionList.scrollTop = nextQuestionListScrollTop;
    }
    window.scrollTo({ top: nextPageScrollTop, behavior: "auto" });
  };

  requestAnimationFrame(restoreScroll);
  setTimeout(restoreScroll, 0);
  setTimeout(restoreScroll, 80);
  setTimeout(restoreScroll, 180);
}

document.addEventListener("submit", (event) => {
  if (event.target?.id === "pathForm") {
    event.preventDefault();
    const input = document.querySelector("#pathInput");
    loadRun(input.value);
    return;
  }

  if (event.target?.id === "searchForm") {
    event.preventDefault();
    state.search = state.searchDraft.trim();
    render();
  }
});

document.addEventListener("input", (event) => {
  if (event.target?.id === "searchInput") {
    state.searchDraft = event.target.value;
  }
});

document.addEventListener("focusin", (event) => {
  if (event.target?.id === "pathInput" && event.target.value === state.runPath) {
    event.target.value = "";
  }
});

document.addEventListener("change", (event) => {
  if (event.target?.id === "pathInput") {
    const selectedPath = event.target.value;
    if (state.availableRuns.some((run) => run.path === selectedPath)) {
      loadRun(selectedPath);
    }
  }
  if (event.target?.id === "correctnessFilter") {
    state.correctnessFilter = event.target.value;
    render();
  }
  if (event.target?.id === "changeFilter") {
    state.changeFilter = event.target.value;
    render();
  }
});

document.addEventListener("click", (event) => {
  const questionButton = event.target.closest("[data-question-key]");
  if (questionButton) {
    const questionList = app.querySelector(".question-list");
    const questionListScrollTop = questionList ? questionList.scrollTop : 0;
    if (questionList) {
      state.questionListScrollTop = questionListScrollTop;
    }
    state.pageScrollTop = window.scrollY;
    state.scrollRestore = {
      questionListScrollTop,
      pageScrollTop: state.pageScrollTop,
    };
    loadQuestion(questionButton.dataset.questionKey);
    return;
  }

  const roundButton = event.target.closest("[data-round-type]");
  if (roundButton) {
    const roundIndex = Number(roundButton.dataset.roundIndex);
    if (roundButton.dataset.roundType === "answer") {
      state.answerRound = roundIndex;
    } else {
      state.reviewRound = roundIndex;
      state.selectedReviewId = "";
    }
    render();
    return;
  }

  const reviewButton = event.target.closest("[data-review-id]");
  if (reviewButton) {
    state.selectedReviewId = reviewButton.dataset.reviewId;
    render();
    return;
  }

  if (event.target.closest("#debugToggle")) {
    state.debugMode = !state.debugMode;
    localStorage.setItem("masViewer.debugMode", String(state.debugMode));
    render();
  }
});

document.addEventListener("mousedown", (event) => {
  const questionButton = event.target.closest("[data-question-key]");
  if (questionButton) {
    event.preventDefault();
  }
});

document.addEventListener("keydown", (event) => {
  const questionButton = event.target.closest("[data-question-key]");
  if (questionButton && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    questionButton.click();
  }
});

document.addEventListener(
  "scroll",
  (event) => {
    if (event.target?.classList?.contains("question-list") && !state.scrollRestore) {
      state.questionListScrollTop = event.target.scrollTop;
    }
    if (event.target === document || event.target === document.documentElement) {
      if (!state.scrollRestore) {
        state.pageScrollTop = window.scrollY;
      }
    }
  },
  true,
);

render();
loadAvailableRuns().finally(() => {
  if (state.runPath) loadRun(state.runPath);
});
