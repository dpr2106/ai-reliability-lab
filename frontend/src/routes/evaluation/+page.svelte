<script lang="ts">
  import { onMount } from 'svelte';
  import { api, ApiError } from '#lib/api';
  import StatusBadge from '#lib/components/StatusBadge.svelte';
  import MetricCard from '#lib/components/MetricCard.svelte';
  import type {
    DatasetSummary,
    DatasetDetail,
    BenchmarkQuery,
    DocumentSummary,
    EvaluationJob,
    RunSummary,
    RunDetail,
    QueryResult,
    RegressionResult,
    RunReproducibilityResponse,
    MetricsMetaResponse
  } from '#lib/types';

  // Navigation tab state
  type TabKey = 'launcher' | 'datasets' | 'runs' | 'failures' | 'regression';
  let activeTab = $state<TabKey>('launcher');

  // Shared state
  let availableDatasets = $state<DatasetSummary[]>([]);
  let availableDocs = $state<DocumentSummary[]>([]);
  let availableRuns = $state<RunSummary[]>([]);
  let metricsCatalog = $state<MetricsMetaResponse | null>(null);
  let isLoading = $state<boolean>(true);
  let errorMessage = $state<string | null>(null);
  let successMessage = $state<string | null>(null);

  // -------------------------------------------------------------------------
  // 1. Experiment Launcher State
  // -------------------------------------------------------------------------
  let launcherExpName = $state<string>('Evaluation Experiment');
  let launcherDatasetId = $state<string>('');
  let launcherSelectedDocIds = $state<string[]>([]);
  let launcherRetrieverType = $state<'bm25' | 'faiss' | 'hybrid'>('hybrid');
  let launcherTopK = $state<number>(5);
  let launcherBm25K1 = $state<number>(1.5);
  let launcherBm25B = $state<number>(0.75);
  let launcherHybridRrfK = $state<number>(60);
  let launcherProvider = $state<'ollama' | 'mock' | 'gemini'>('ollama');
  let launcherModelName = $state<string>('llama3.2');
  let launcherTemperature = $state<number>(0.0);
  let launcherMaxTokens = $state<number>(1024);
  let launcherPromptVersion = $state<string>('v1');
  let launcherEnableGroundedness = $state<boolean>(true);
  let launcherEnableCitations = $state<boolean>(true);
  let launcherEnableNli = $state<boolean>(true);
  let launcherEnableJudge = $state<boolean>(false);

  // Active Job State
  let activeJob = $state<EvaluationJob | null>(null);
  let isLaunching = $state<boolean>(false);
  let jobPollInterval: ReturnType<typeof setInterval> | null = null;

  // -------------------------------------------------------------------------
  // 2. Dataset Studio State
  // -------------------------------------------------------------------------
  let selectedDataset = $state<DatasetDetail | null>(null);
  let isDatasetLoading = $state<boolean>(false);
  let showCreateDatasetModal = $state<boolean>(false);
  let showImportModal = $state<boolean>(false);
  let showAddQueryModal = $state<boolean>(false);

  // New dataset form
  let newDatasetName = $state<string>('');
  let newDatasetDesc = $state<string>('');
  let newDatasetScopedDocs = $state<string[]>([]);

  // Import dataset form
  let importName = $state<string>('');
  let importJsonlContent = $state<string>('');
  let importDesc = $state<string>('');

  // Add query form
  let newQueryId = $state<string>('');
  let newQueryText = $state<string>('');
  let newQueryAnswer = $state<string>('');
  let newQueryChunks = $state<string>('');
  let newQueryCitations = $state<string>('');

  // -------------------------------------------------------------------------
  // 3. Failure Explorer State
  // -------------------------------------------------------------------------
  let failureRunId = $state<string>('');
  let failureCategory = $state<string>('all');
  let failureSortBy = $state<string>('default');
  let failureQueries = $state<QueryResult[]>([]);
  let isFailureLoading = $state<boolean>(false);
  let expandedQueryId = $state<string | null>(null);

  // -------------------------------------------------------------------------
  // 4. Regression Studio State
  // -------------------------------------------------------------------------
  let regressionBaselineId = $state<string>('');
  let regressionCandidateId = $state<string>('');
  let regressionTolerance = $state<number>(0.02);
  let regressionResult = $state<RegressionResult | null>(null);
  let isComparing = $state<boolean>(false);

  // -------------------------------------------------------------------------
  // Reproducibility modal
  // -------------------------------------------------------------------------
  let selectedReproducibility = $state<RunReproducibilityResponse | null>(null);
  let showReproducibilityModal = $state<boolean>(false);

  // -------------------------------------------------------------------------
  // Lifecycle & Initial Load
  // -------------------------------------------------------------------------
  async function loadInitialData() {
    isLoading = true;
    errorMessage = null;
    try {
      const [dsRes, docRes, runRes, metaRes] = await Promise.all([
        api.listDatasets({ limit: 100 }),
        api.listDocuments({ limit: 100 }),
        api.listRuns({ limit: 50 }),
        api.getMetricsMeta()
      ]);
      availableDatasets = dsRes.datasets;
      availableDocs = docRes.items.filter((d) => d.status === 'ready');
      availableRuns = runRes.runs;
      metricsCatalog = metaRes;

      if (availableDatasets.length > 0 && !launcherDatasetId) {
        launcherDatasetId = availableDatasets[0].dataset_id;
      }
      if (availableRuns.length > 0 && !failureRunId) {
        failureRunId = availableRuns[0].run_id;
      }
      if (availableRuns.length >= 2 && !regressionBaselineId && !regressionCandidateId) {
        regressionBaselineId = availableRuns[1].run_id;
        regressionCandidateId = availableRuns[0].run_id;
      }
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      isLoading = false;
    }
  }

  onMount(() => {
    loadInitialData();
    return () => {
      if (jobPollInterval) clearInterval(jobPollInterval);
    };
  });

  // -------------------------------------------------------------------------
  // Launcher Actions
  // -------------------------------------------------------------------------
  async function launchExperiment() {
    if (!launcherDatasetId) {
      errorMessage = 'Please select a benchmark dataset before launching.';
      return;
    }
    if (!launcherExpName.trim()) {
      errorMessage = 'Please enter an experiment name.';
      return;
    }

    isLaunching = true;
    errorMessage = null;
    successMessage = null;

    try {
      const job = await api.launchEvaluation({
        experiment_name: launcherExpName.trim(),
        dataset_id: launcherDatasetId,
        document_ids: launcherSelectedDocIds.length > 0 ? launcherSelectedDocIds : undefined,
        retriever_type: launcherRetrieverType,
        retriever_params: {
          k1: launcherBm25K1,
          b: launcherBm25B,
          rrf_k: launcherHybridRrfK
        },
        top_k: launcherTopK,
        provider: launcherProvider,
        model_name: launcherModelName.trim() || null,
        temperature: launcherTemperature,
        max_output_tokens: launcherMaxTokens,
        prompt_version: launcherPromptVersion,
        enable_groundedness: launcherEnableGroundedness,
        enable_citations: launcherEnableCitations,
        enable_nli: launcherEnableNli,
        enable_judge: launcherEnableJudge
      });

      activeJob = job;
      startJobPolling(job.job_id);
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
      isLaunching = false;
    }
  }

  function startJobPolling(jobId: string) {
    if (jobPollInterval) clearInterval(jobPollInterval);
    jobPollInterval = setInterval(async () => {
      try {
        const updated = await api.getEvaluationJob(jobId);
        activeJob = updated;
        if (updated.status === 'completed') {
          if (jobPollInterval) clearInterval(jobPollInterval);
          isLaunching = false;
          successMessage = `Evaluation run '${updated.experiment_name}' completed successfully! Run ID: ${updated.run_id}`;
          // Refresh runs list
          const runRes = await api.listRuns({ limit: 50 });
          availableRuns = runRes.runs;
        } else if (updated.status === 'failed') {
          if (jobPollInterval) clearInterval(jobPollInterval);
          isLaunching = false;
          errorMessage = `Evaluation failed: ${updated.error_message ?? 'Unknown error'}`;
        }
      } catch {
        // Polling failure, retry on next tick
      }
    }, 1500);
  }

  // -------------------------------------------------------------------------
  // Dataset Actions
  // -------------------------------------------------------------------------
  async function selectDataset(datasetId: string) {
    isDatasetLoading = true;
    try {
      selectedDataset = await api.getDataset(datasetId);
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      isDatasetLoading = false;
    }
  }

  async function handleCreateDataset() {
    if (!newDatasetName.trim()) {
      errorMessage = 'Dataset name is required.';
      return;
    }
    try {
      const created = await api.createDataset({
        name: newDatasetName.trim(),
        description: newDatasetDesc.trim() || null,
        document_scope: newDatasetScopedDocs,
        queries: []
      });
      showCreateDatasetModal = false;
      newDatasetName = '';
      newDatasetDesc = '';
      newDatasetScopedDocs = [];
      const dsRes = await api.listDatasets({ limit: 100 });
      availableDatasets = dsRes.datasets;
      selectDataset(created.dataset_id);
      successMessage = `Created benchmark dataset '${created.name}'`;
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }

  async function handleImportDataset() {
    if (!importName.trim() || !importJsonlContent.trim()) {
      errorMessage = 'Dataset name and JSONL content are required.';
      return;
    }
    try {
      const imported = await api.importDataset({
        name: importName.trim(),
        jsonl_content: importJsonlContent.trim(),
        description: importDesc.trim() || null
      });
      showImportModal = false;
      importName = '';
      importJsonlContent = '';
      importDesc = '';
      const dsRes = await api.listDatasets({ limit: 100 });
      availableDatasets = dsRes.datasets;
      selectDataset(imported.dataset_id);
      successMessage = `Imported dataset '${imported.name}' (${imported.query_count} queries)`;
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }

  async function handleExportDataset(datasetId: string, name: string) {
    try {
      const jsonlStr = await api.exportDataset(datasetId);
      const blob = new Blob([jsonlStr], { type: 'application/x-jsonlines' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dataset_${datasetId}.jsonl`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }

  async function handleAddQuery() {
    if (!selectedDataset) return;
    if (!newQueryId.trim() || !newQueryText.trim()) {
      errorMessage = 'Query ID and text are required.';
      return;
    }
    try {
      const cids = newQueryChunks
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
      const grades: Record<string, number> = {};
      cids.forEach((c) => (grades[c] = 2));

      const reqCits = newQueryCitations
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);

      await api.addQuery(selectedDataset.dataset_id, {
        query_id: newQueryId.trim(),
        text: newQueryText.trim(),
        ground_truth_answer: newQueryAnswer.trim() || null,
        relevant_chunk_ids: cids,
        relevance_grades: grades,
        required_citations: reqCits
      });

      showAddQueryModal = false;
      newQueryId = '';
      newQueryText = '';
      newQueryAnswer = '';
      newQueryChunks = '';
      newQueryCitations = '';
      await selectDataset(selectedDataset.dataset_id);
      successMessage = 'Benchmark query added successfully.';
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }

  async function handleDeleteQuery(queryId: string) {
    if (!selectedDataset) return;
    if (!confirm(`Delete query '${queryId}'?`)) return;
    try {
      await api.deleteQuery(selectedDataset.dataset_id, queryId);
      await selectDataset(selectedDataset.dataset_id);
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }

  // -------------------------------------------------------------------------
  // Failure Explorer Actions
  // -------------------------------------------------------------------------
  async function loadFailureQueries() {
    if (!failureRunId) return;
    isFailureLoading = true;
    errorMessage = null;
    try {
      const res = await api.getRunQueries(failureRunId, {
        limit: 100,
        failed_only: failureCategory !== 'all' && failureCategory !== 'passed',
        failure_type: failureCategory !== 'all' ? failureCategory : undefined,
        sort_by: failureSortBy
      });
      failureQueries = res.queries;
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      isFailureLoading = false;
    }
  }

  // -------------------------------------------------------------------------
  // Regression Actions
  // -------------------------------------------------------------------------
  async function executeRegressionComparison() {
    if (!regressionBaselineId || !regressionCandidateId) {
      errorMessage = 'Please select both a baseline and a candidate run.';
      return;
    }
    if (regressionBaselineId === regressionCandidateId) {
      errorMessage = 'Baseline and candidate runs must be distinct.';
      return;
    }

    isComparing = true;
    errorMessage = null;
    regressionResult = null;
    try {
      regressionResult = await api.compareRuns({
        baseline_run_id: regressionBaselineId,
        candidate_run_id: regressionCandidateId,
        tolerance: regressionTolerance
      });
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      isComparing = false;
    }
  }

  // -------------------------------------------------------------------------
  // Reproducibility Actions
  // -------------------------------------------------------------------------
  async function inspectReproducibility(runId: string) {
    try {
      selectedReproducibility = await api.getRunReproducibility(runId);
      showReproducibilityModal = true;
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }

  async function toggleBaseline(runId: string, current: boolean) {
    try {
      await api.setRunBaseline(runId, !current);
      const runRes = await api.listRuns({ limit: 50 });
      availableRuns = runRes.runs;
      successMessage = !current ? `Run '${runId}' set as baseline.` : `Run '${runId}' baseline removed.`;
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }
</script>

<div class="studio-wrapper">
  <!-- Studio Header -->
  <header class="studio-header">
    <div class="header-left">
      <div class="title-row">
        <h1 class="studio-title">RAG Evaluation, Comparison & Regression Studio</h1>
        <span class="version-tag">V2-005</span>
      </div>
      <p class="studio-subtitle">
        Systematic local RAG evaluation, golden benchmark datasets, query failure diagnostics, and deterministic regression detection.
      </p>
    </div>

    <!-- Navigation Pills -->
    <nav class="studio-tabs" aria-label="Evaluation Studio Tabs">
      <button
        class="tab-btn"
        class:active={activeTab === 'launcher'}
        onclick={() => (activeTab = 'launcher')}
      >
        <span class="tab-icon">⚡</span>
        Experiment Launcher
      </button>
      <button
        class="tab-btn"
        class:active={activeTab === 'datasets'}
        onclick={() => {
          activeTab = 'datasets';
          if (!selectedDataset && availableDatasets.length > 0) {
            selectDataset(availableDatasets[0].dataset_id);
          }
        }}
      >
        <span class="tab-icon">📚</span>
        Datasets Studio ({availableDatasets.length})
      </button>
      <button
        class="tab-btn"
        class:active={activeTab === 'runs'}
        onclick={() => (activeTab = 'runs')}
      >
        <span class="tab-icon">📊</span>
        Runs & Baselines ({availableRuns.length})
      </button>
      <button
        class="tab-btn"
        class:active={activeTab === 'failures'}
        onclick={() => {
          activeTab = 'failures';
          loadFailureQueries();
        }}
      >
        <span class="tab-icon">🔍</span>
        Failure Explorer
      </button>
      <button
        class="tab-btn"
        class:active={activeTab === 'regression'}
        onclick={() => (activeTab = 'regression')}
      >
        <span class="tab-icon">⚖️</span>
        Regression Studio
      </button>
    </nav>
  </header>

  <!-- Global Feedback Banners -->
  {#if errorMessage}
    <div class="banner banner-error">
      <span class="banner-icon">⚠️</span>
      <div class="banner-text">{errorMessage}</div>
      <button class="banner-close" onclick={() => (errorMessage = null)}>✕</button>
    </div>
  {/if}

  {#if successMessage}
    <div class="banner banner-success">
      <span class="banner-icon">✓</span>
      <div class="banner-text">{successMessage}</div>
      <button class="banner-close" onclick={() => (successMessage = null)}>✕</button>
    </div>
  {/if}

  <!-- TAB 1: EXPERIMENT LAUNCHER -->
  {#if activeTab === 'launcher'}
    <div class="tab-pane launcher-pane">
      <div class="launcher-grid">
        <!-- Configuration Column -->
        <div class="card config-card">
          <div class="card-header">
            <h2 class="card-title">1. Experiment & Corpus Configuration</h2>
            <span class="badge badge-neutral">Step 1 of 2</span>
          </div>

          <div class="form-group">
            <label for="exp-name" class="form-label">Experiment Name</label>
            <input
              id="exp-name"
              type="text"
              class="form-input"
              bind:value={launcherExpName}
              placeholder="e.g., BM25 vs Hybrid RRF Baseline"
            />
          </div>

          <div class="form-group">
            <label for="dataset-select" class="form-label">Benchmark Evaluation Dataset</label>
            <select id="dataset-select" class="form-select" bind:value={launcherDatasetId}>
              {#each availableDatasets as ds}
                <option value={ds.dataset_id}>
                  {ds.name} ({ds.query_count} queries){ds.is_synthetic ? ' [SYNTHETIC]' : ''}
                </option>
              {/each}
            </select>
            <span class="form-hint">
              Ground-truth answers and relevance labels are strictly isolated and never leak into generation prompts.
            </span>
          </div>

          <div class="form-group">
            <span class="form-label">
              Document Corpus Scope ({availableDocs.length} ready documents)
            </span>
            <div class="doc-checklist">
              {#each availableDocs as doc}
                <label class="doc-check-item">
                  <input
                    type="checkbox"
                    checked={launcherSelectedDocIds.includes(doc.doc_id)}
                    onchange={() => {
                      if (launcherSelectedDocIds.includes(doc.doc_id)) {
                        launcherSelectedDocIds = launcherSelectedDocIds.filter((id) => id !== doc.doc_id);
                      } else {
                        launcherSelectedDocIds = [...launcherSelectedDocIds, doc.doc_id];
                      }
                    }}
                  />
                  <span class="doc-title">{doc.filename}</span>
                  <span class="doc-meta">{doc.chunk_count} chunks</span>
                </label>
              {/each}
            </div>
            <div class="check-actions">
              <button
                type="button"
                class="btn-text"
                onclick={() => (launcherSelectedDocIds = availableDocs.map((d) => d.doc_id))}
              >
                Select All
              </button>
              <button
                type="button"
                class="btn-text"
                onclick={() => (launcherSelectedDocIds = [])}
              >
                Clear (Evaluate Entire Repository)
              </button>
            </div>
          </div>
        </div>

        <!-- Pipeline & Generation Column -->
        <div class="card pipeline-card">
          <div class="card-header">
            <h2 class="card-title">2. Retrieval & Generation Pipeline</h2>
            <span class="badge badge-neutral">Step 2 of 2</span>
          </div>

          <!-- Retrieval Parameters -->
          <div class="form-row">
            <div class="form-group col-6">
              <label for="retriever-type" class="form-label">Retrieval Engine</label>
              <select id="retriever-type" class="form-select" bind:value={launcherRetrieverType}>
                <option value="hybrid">Hybrid RRF (BM25 + FAISS)</option>
                <option value="bm25">BM25 (Lexical Index)</option>
                <option value="faiss">FAISS Flat (Dense Vectors)</option>
              </select>
            </div>

            <div class="form-group col-6">
              <label for="top-k" class="form-label">Top-K Cutoff: {launcherTopK}</label>
              <input
                id="top-k"
                type="range"
                min="1"
                max="20"
                class="form-range"
                bind:value={launcherTopK}
              />
            </div>
          </div>

          {#if launcherRetrieverType === 'hybrid' || launcherRetrieverType === 'bm25'}
            <div class="form-row param-row">
              <div class="form-group col-4">
                <label for="k1-param" class="form-label">BM25 k1: {launcherBm25K1}</label>
                <input
                  id="k1-param"
                  type="number"
                  step="0.1"
                  min="0.1"
                  max="5.0"
                  class="form-input-sm"
                  bind:value={launcherBm25K1}
                />
              </div>
              <div class="form-group col-4">
                <label for="b-param" class="form-label">BM25 b: {launcherBm25B}</label>
                <input
                  id="b-param"
                  type="number"
                  step="0.05"
                  min="0.0"
                  max="1.0"
                  class="form-input-sm"
                  bind:value={launcherBm25B}
                />
              </div>
              {#if launcherRetrieverType === 'hybrid'}
                <div class="form-group col-4">
                  <label for="rrf-param" class="form-label">RRF Constant k: {launcherHybridRrfK}</label>
                  <input
                    id="rrf-param"
                    type="number"
                    min="1"
                    max="200"
                    class="form-input-sm"
                    bind:value={launcherHybridRrfK}
                  />
                </div>
              {/if}
            </div>
          {/if}

          <!-- Generation Parameters -->
          <div class="form-row">
            <div class="form-group col-6">
              <label for="provider-select" class="form-label">Inference Provider</label>
              <select id="provider-select" class="form-select" bind:value={launcherProvider}>
                <option value="ollama">Ollama (Local Inference)</option>
                <option value="mock">Mock LLM (Deterministic Test)</option>
                <option value="gemini">Gemini API (Cloud Provider)</option>
              </select>
            </div>

            <div class="form-group col-6">
              <label for="model-name" class="form-label">Model Name</label>
              <input
                id="model-name"
                type="text"
                class="form-input"
                bind:value={launcherModelName}
                placeholder="e.g. llama3.2, mistral:instruct"
              />
            </div>
          </div>

          <!-- Evaluation Diagnostics Toggles -->
          <div class="form-group">
            <span class="form-label">Active Diagnostic Evaluation Stages</span>
            <div class="stages-grid">
              <label class="stage-toggle">
                <input type="checkbox" bind:checked={launcherEnableGroundedness} />
                <span class="stage-label">Lexical Groundedness (N-gram & LCS)</span>
              </label>
              <label class="stage-toggle">
                <input type="checkbox" bind:checked={launcherEnableCitations} />
                <span class="stage-label">Citation Audit (Precision & Recall)</span>
              </label>
              <label class="stage-toggle">
                <input type="checkbox" bind:checked={launcherEnableNli} />
                <span class="stage-label">NLI Entailment (DeBERTa cross-encoder)</span>
              </label>
              <label class="stage-toggle">
                <input type="checkbox" bind:checked={launcherEnableJudge} />
                <span class="stage-label">LLM-as-a-Judge Faithfulness Audit</span>
              </label>
            </div>
          </div>

          <!-- Provenance Notice -->
          {#if launcherProvider === 'mock'}
            <div class="provenance-banner warning-prov">
              <span class="prov-badge">MOCK PROVENANCE</span>
              <span>
                Generation results will be simulated deterministically without external model calls. Mock evaluations must not be presented as live model accuracy.
              </span>
            </div>
          {:else if launcherProvider === 'ollama'}
            <div class="provenance-banner local-prov">
              <span class="prov-badge">LOCAL FIRST (OLLAMA)</span>
              <span>
                Zero data egress. Evaluates locally on your hardware. Ensure Ollama daemon is active.
              </span>
            </div>
          {/if}

          <!-- Live Progress / Launch Action -->
          <div class="launch-footer">
            {#if isLaunching && activeJob}
              <div class="job-progress-box">
                <div class="progress-header">
                  <span class="progress-title">Executing: {activeJob.experiment_name}</span>
                  <span class="progress-stat">
                    {activeJob.queries_completed} / {activeJob.total_queries} queries ({activeJob.progress_percent}%)
                  </span>
                </div>
                <div class="progress-track">
                  <div
                    class="progress-fill"
                    style="width: {activeJob.progress_percent}%"
                  ></div>
                </div>
              </div>
            {/if}

            <button
              class="btn-primary launch-btn"
              disabled={isLaunching}
              onclick={launchExperiment}
            >
              {#if isLaunching}
                <span class="spinner"></span>
                Evaluating Benchmark...
              {:else}
                Launch Evaluation Experiment 🚀
              {/if}
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- TAB 2: DATASET STUDIO -->
  {#if activeTab === 'datasets'}
    <div class="tab-pane datasets-pane">
      <div class="dataset-layout">
        <!-- Dataset Sidebar -->
        <aside class="dataset-sidebar">
          <div class="sidebar-header">
            <h3 class="sidebar-title">Benchmark Datasets</h3>
            <div class="sidebar-actions">
              <button class="btn-sm btn-outline" onclick={() => (showImportModal = true)}>
                Import JSONL
              </button>
              <button class="btn-sm btn-primary" onclick={() => (showCreateDatasetModal = true)}>
                + New
              </button>
            </div>
          </div>

          <div class="dataset-list">
            {#each availableDatasets as ds}
              <button
                class="dataset-item-btn"
                class:selected={selectedDataset?.dataset_id === ds.dataset_id}
                onclick={() => selectDataset(ds.dataset_id)}
              >
                <div class="ds-item-title-row">
                  <span class="ds-item-name">{ds.name}</span>
                  {#if ds.is_synthetic}
                    <span class="badge badge-synthetic">SYNTHETIC</span>
                  {/if}
                </div>
                <div class="ds-item-meta">
                  <span>{ds.query_count} queries</span>
                  <span>{ds.document_scope.length > 0 ? `${ds.document_scope.length} docs scoped` : 'All docs'}</span>
                </div>
              </button>
            {/each}
          </div>
        </aside>

        <!-- Dataset Content / Query Table -->
        <main class="dataset-main">
          {#if isDatasetLoading}
            <div class="loading-state">Loading dataset questions...</div>
          {:else if selectedDataset}
            <div class="dataset-header-card card">
              <div class="dh-top">
                <div>
                  <h2 class="dh-title">{selectedDataset.name}</h2>
                  <p class="dh-desc">{selectedDataset.description ?? 'No description provided.'}</p>
                </div>
                <div class="dh-actions">
                  <button
                    class="btn-sm btn-outline"
                    onclick={() => handleExportDataset(selectedDataset!.dataset_id, selectedDataset!.name)}
                  >
                    Export JSONL 📥
                  </button>
                  <button
                    class="btn-sm btn-primary"
                    onclick={() => (showAddQueryModal = true)}
                  >
                    + Add Query
                  </button>
                </div>
              </div>

              <div class="dh-tags">
                <span class="tag">ID: <code>{selectedDataset.dataset_id}</code></span>
                <span class="tag">Queries: <strong>{selectedDataset.query_count}</strong></span>
                {#if selectedDataset.is_synthetic}
                  <span class="badge badge-synthetic">Demonstration Synthetic Benchmark</span>
                {/if}
              </div>
            </div>

            <!-- Queries Table -->
            <div class="queries-card card">
              <h3 class="card-title">Golden Benchmark Queries ({selectedDataset.queries.length})</h3>

              {#if selectedDataset.queries.length === 0}
                <div class="empty-queries">
                  <p>No queries in this benchmark yet.</p>
                  <button class="btn-sm btn-primary" onclick={() => (showAddQueryModal = true)}>
                    Add First Query
                  </button>
                </div>
              {:else}
                <div class="table-container">
                  <table class="data-table">
                    <thead>
                      <tr>
                        <th>Query ID</th>
                        <th>Question / Prompt</th>
                        <th>Relevant Gold Chunks</th>
                        <th>Ground Truth Answer</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {#each selectedDataset.queries as q}
                        <tr>
                          <td><code>{q.query_id}</code></td>
                          <td class="query-cell">{q.text}</td>
                          <td>
                            <div class="chunks-pill-group">
                              {#each q.relevant_chunk_ids as cid}
                                <span class="chunk-pill" title="Grade: {q.relevance_grades[cid] ?? 1}">
                                  {cid} (G:{q.relevance_grades[cid] ?? 1})
                                </span>
                              {/each}
                            </div>
                          </td>
                          <td class="answer-cell">
                            {q.ground_truth_answer ?? '—'}
                          </td>
                          <td>
                            <button
                              class="btn-danger-sm"
                              onclick={() => handleDeleteQuery(q.query_id)}
                              title="Delete query"
                            >
                              ✕
                            </button>
                          </td>
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              {/if}
            </div>
          {:else}
            <div class="empty-state">Select a benchmark dataset to inspect its golden queries.</div>
          {/if}
        </main>
      </div>
    </div>
  {/if}

  <!-- TAB 3: RUNS & BASELINES -->
  {#if activeTab === 'runs'}
    <div class="tab-pane runs-pane">
      <div class="runs-header card">
        <h2 class="card-title">Historical Evaluation Runs ({availableRuns.length})</h2>
        <p class="text-muted">
          Persisted experiments retain complete reproducibility manifests, configuration digests, and provenance metadata.
        </p>
      </div>

      <div class="runs-grid">
        {#each availableRuns as run}
          {@const isBaseline = Boolean(run.metadata?.is_baseline)}
          {@const provType = run.metadata?.provenance_type || (run.metadata?.is_mock ? 'mock' : 'local')}
          <div class="run-card card" class:is-baseline-card={isBaseline}>
            <div class="rc-header">
              <div>
                <span class="rc-id"><code>{run.run_id}</code></span>
                <h3 class="rc-name">{run.experiment_name}</h3>
              </div>
              <div class="rc-badges">
                {#if isBaseline}
                  <span class="badge badge-baseline">BASELINE</span>
                {/if}
                {#if provType === 'mock'}
                  <span class="badge badge-prov-mock">MOCK</span>
                {:else if provType === 'local'}
                  <span class="badge badge-prov-local">LOCAL</span>
                {:else}
                  <span class="badge badge-prov-cloud">CLOUD</span>
                {/if}
              </div>
            </div>

            <!-- Key Metric Pills -->
            <div class="rc-metrics">
              <div class="metric-pill">
                <span class="mp-label">Recall@K</span>
                <span class="mp-val">
                  {run.metrics['recall@5'] !== undefined
                    ? `${(run.metrics['recall@5'] * 100).toFixed(1)}%`
                    : run.metrics['mean_recall@5'] !== undefined
                    ? `${(run.metrics['mean_recall@5'] * 100).toFixed(1)}%`
                    : 'N/A'}
                </span>
              </div>
              <div class="metric-pill">
                <span class="mp-label">HitRate@K</span>
                <span class="mp-val">
                  {run.metrics['hit_rate@5'] !== undefined
                    ? `${(run.metrics['hit_rate@5'] * 100).toFixed(1)}%`
                    : 'N/A'}
                </span>
              </div>
              <div class="metric-pill">
                <span class="mp-label">NDCG@K</span>
                <span class="mp-val">
                  {run.metrics['ndcg@5'] !== undefined ? run.metrics['ndcg@5'].toFixed(3) : 'N/A'}
                </span>
              </div>
              <div class="metric-pill">
                <span class="mp-label">Queries</span>
                <span class="mp-val">{run.query_count}</span>
              </div>
            </div>

            <div class="rc-footer">
              <span class="rc-time">{new Date(run.created_at).toLocaleString()}</span>
              <div class="rc-actions">
                <button
                  class="btn-sm btn-outline"
                  onclick={() => toggleBaseline(run.run_id, isBaseline)}
                >
                  {isBaseline ? 'Unset Baseline' : 'Set as Baseline'}
                </button>
                <button
                  class="btn-sm btn-outline"
                  onclick={() => inspectReproducibility(run.run_id)}
                >
                  Provenance 🔍
                </button>
                <button
                  class="btn-sm btn-primary"
                  onclick={() => {
                    failureRunId = run.run_id;
                    activeTab = 'failures';
                    loadFailureQueries();
                  }}
                >
                  Inspect Failures ⚡
                </button>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- TAB 4: FAILURE EXPLORER -->
  {#if activeTab === 'failures'}
    <div class="tab-pane failures-pane">
      <div class="filters-bar card">
        <div class="form-row">
          <div class="form-group col-4">
            <label for="fail-run" class="form-label">Inspected Run</label>
            <select
              id="fail-run"
              class="form-select"
              bind:value={failureRunId}
              onchange={loadFailureQueries}
            >
              {#each availableRuns as r}
                <option value={r.run_id}>{r.experiment_name} ({r.run_id})</option>
              {/each}
            </select>
          </div>

          <div class="form-group col-4">
            <label for="fail-category" class="form-label">Failure Classification Mode</label>
            <select
              id="fail-category"
              class="form-select"
              bind:value={failureCategory}
              onchange={loadFailureQueries}
            >
              <option value="all">All Evaluated Queries</option>
              <option value="retrieval_miss">Complete Retrieval Miss (HitRate@K = 0)</option>
              <option value="partial_retrieval">Partial Retrieval Miss (Recall@K &lt; 1.0)</option>
              <option value="citation_hallucination">Citation Hallucination (Precision &lt; 1.0)</option>
              <option value="citation_omission">Citation Omission (Recall &lt; 1.0)</option>
              <option value="ungrounded_generation">Ungrounded Generation</option>
              <option value="passed">Clean Passed Queries</option>
            </select>
          </div>

          <div class="form-group col-4">
            <label for="fail-sort" class="form-label">Sort By</label>
            <select
              id="fail-sort"
              class="form-select"
              bind:value={failureSortBy}
              onchange={loadFailureQueries}
            >
              <option value="default">Default (Query Order)</option>
              <option value="worst_hitrate">Lowest Hit Rate First</option>
              <option value="worst_recall">Lowest Recall First</option>
              <option value="latency">Highest Latency First</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Failure Results List -->
      {#if isFailureLoading}
        <div class="loading-state">Auditing query failure modes...</div>
      {:else if failureQueries.length === 0}
        <div class="empty-state">No queries match the selected filter criteria.</div>
      {:else}
        <div class="failures-list">
          {#each failureQueries as q}
            {@const diag = q.failure_diagnostic}
            {@const isExpanded = expandedQueryId === q.query_id}
            <div class="failure-card card" class:has-defect={diag?.has_failure}>
              <div
                class="fc-header"
                role="button"
                tabindex="0"
                onclick={() => (expandedQueryId = isExpanded ? null : q.query_id)}
                onkeydown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    expandedQueryId = isExpanded ? null : q.query_id;
                  }
                }}
              >
                <div class="fc-title-col">
                  <div class="fc-id-row">
                    <span class="q-id"><code>{q.query_id}</code></span>
                    {#if diag?.classification === 'retrieval_miss'}
                      <span class="badge badge-danger">RETRIEVAL MISS</span>
                    {:else if diag?.classification === 'partial_retrieval'}
                      <span class="badge badge-warning">PARTIAL RECALL</span>
                    {:else if diag?.classification === 'citation_hallucination'}
                      <span class="badge badge-danger">CITATION HALLUCINATION</span>
                    {:else if diag?.classification === 'citation_omission'}
                      <span class="badge badge-warning">CITATION OMISSION</span>
                    {:else if diag?.classification === 'ungrounded_generation'}
                      <span class="badge badge-danger">UNGROUNDED</span>
                    {:else}
                      <span class="badge badge-success">PASSED</span>
                    {/if}
                    <span class="fc-latency">{q.latency_ms?.toFixed(1) ?? '—'} ms</span>
                  </div>
                  <h4 class="fc-query-text">{q.text ?? 'No query text recorded'}</h4>
                  <p class="fc-explanation">{diag?.explanation ?? q.failure_trace ?? 'Query evaluated without defects.'}</p>
                </div>
                <div class="fc-expand-indicator">
                  {isExpanded ? '▲' : '▼'}
                </div>
              </div>

              {#if isExpanded}
                <div class="fc-expanded-body">
                  <div class="fc-side-by-side">
                    <!-- Left: Gold Evidence -->
                    <div class="f-subcard">
                      <h5 class="sub-title">Gold-Standard Evidence Required</h5>
                      {#if q.relevant_chunk_ids && q.relevant_chunk_ids.length > 0}
                        <div class="chunks-list">
                          {#each q.relevant_chunk_ids as rcid}
                            <div class="chunk-ref-box">
                              <span class="cr-id"><code>{rcid}</code></span>
                              <span class="cr-grade">Grade: {q.relevance_grades?.[rcid] ?? 1}</span>
                            </div>
                          {/each}
                        </div>
                      {:else}
                        <p class="text-muted">No explicit gold chunk references provided.</p>
                      {/if}

                      <h5 class="sub-title mt-3">Expected Reference Answer</h5>
                      <p class="ref-answer-text">{q.ground_truth_answer ?? 'No ground-truth reference answer stored.'}</p>
                    </div>

                    <!-- Right: Retrieved & Generated -->
                    <div class="f-subcard">
                      <h5 class="sub-title">Retrieved Evidence Candidates</h5>
                      {#if q.candidates && q.candidates.length > 0}
                        <div class="candidates-list">
                          {#each q.candidates as cand, idx}
                            <div class="cand-item" class:is-relevant={q.relevant_chunk_ids?.includes(cand.chunk_id)}>
                              <div class="ci-header">
                                <span>Rank #{idx + 1}: <code>{cand.chunk_id}</code></span>
                                <span class="ci-score">Score: {cand.score?.toFixed(4) ?? 'N/A'}</span>
                              </div>
                              <p class="ci-content">{cand.content ? cand.content.slice(0, 180) + '...' : 'No snippet'}</p>
                            </div>
                          {/each}
                        </div>
                      {:else}
                        <p class="text-muted">No candidates recorded for this query turn.</p>
                      {/if}

                      <h5 class="sub-title mt-3">Generated Answer</h5>
                      <div class="gen-answer-box">
                        {q.generated_answer ?? q.answer ?? 'Generation stage disabled or no answer produced.'}
                      </div>
                    </div>
                  </div>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  <!-- TAB 5: REGRESSION STUDIO -->
  {#if activeTab === 'regression'}
    <div class="tab-pane regression-pane">
      <div class="card compare-setup-card">
        <h2 class="card-title">Regression Comparison Configuration</h2>
        <div class="form-row">
          <div class="form-group col-5">
            <label for="reg-base" class="form-label">Baseline Reference Run</label>
            <select id="reg-base" class="form-select" bind:value={regressionBaselineId}>
              {#each availableRuns as r}
                <option value={r.run_id}>
                  {r.experiment_name} ({r.run_id}){r.metadata?.is_baseline ? ' ★ [BASELINE]' : ''}
                </option>
              {/each}
            </select>
          </div>

          <div class="form-group col-5">
            <label for="reg-cand" class="form-label">Candidate Run to Evaluate</label>
            <select id="reg-cand" class="form-select" bind:value={regressionCandidateId}>
              {#each availableRuns as r}
                <option value={r.run_id}>{r.experiment_name} ({r.run_id})</option>
              {/each}
            </select>
          </div>

          <div class="form-group col-2">
            <label for="reg-tol" class="form-label">Tolerance: {regressionTolerance}</label>
            <input
              id="reg-tol"
              type="number"
              step="0.01"
              min="0.0"
              max="0.5"
              class="form-input"
              bind:value={regressionTolerance}
            />
          </div>
        </div>

        <button
          class="btn-primary mt-2"
          disabled={isComparing}
          onclick={executeRegressionComparison}
        >
          {isComparing ? 'Comparing...' : 'Run Regression Analysis ⚖️'}
        </button>
      </div>

      {#if regressionResult}
        <div class="regression-report">
          <!-- Verdict Banner -->
          <div
            class="verdict-banner"
            class:v-pass={regressionResult.status === 'PASS'}
            class:v-regression={regressionResult.status === 'REGRESSION'}
            class:v-incomplete={regressionResult.status === 'INCOMPLETE'}
          >
            <div class="verdict-label">
              VERDICT: <strong>{regressionResult.status}</strong>
            </div>
            <div class="verdict-details">
              {#if regressionResult.status === 'PASS'}
                Candidate meets or exceeds all baseline metrics within absolute tolerance ({regressionTolerance}).
              {:else if regressionResult.status === 'REGRESSION'}
                Candidate degraded beyond tolerance on {regressionResult.regressed_aggregate_metrics.length} aggregate metric(s) and {regressionResult.regressed_query_ids.length} individual query/queries.
              {:else}
                Comparison is incomplete. Distinct query sets or incomparable configurations detected.
              {/if}
            </div>
          </div>

          {#if regressionResult.warnings && regressionResult.warnings.length > 0}
            <div class="warnings-box">
              <span class="wb-title">Comparison Incompatibilities / Warnings:</span>
              <ul>
                {#each regressionResult.warnings as w}
                  <li>{w}</li>
                {/each}
              </ul>
            </div>
          {/if}

          <!-- Aggregate Metric Comparison Table -->
          <div class="card mt-3">
            <h3 class="card-title">Aggregate Metric Deltas</h3>
            <div class="table-container">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Candidate</th>
                    <th>Delta</th>
                    <th>Degradation</th>
                    <th>Tolerance</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {#each Object.entries(regressionResult.aggregate_deltas) as [mName, d]}
                    <tr class:regressed-row={d.is_regressed}>
                      <td><strong>{mName}</strong></td>
                      <td>{d.baseline_score.toFixed(4)}</td>
                      <td>{d.candidate_score.toFixed(4)}</td>
                      <td class={d.delta >= 0 ? 'text-green' : 'text-red'}>
                        {d.delta >= 0 ? `+${d.delta.toFixed(4)}` : d.delta.toFixed(4)}
                      </td>
                      <td>{d.degradation > 0 ? d.degradation.toFixed(4) : '0.0000'}</td>
                      <td>{d.tolerance}</td>
                      <td>
                        {#if d.is_regressed}
                          <span class="badge badge-danger">REGRESSED</span>
                        {:else if d.delta > 0.001}
                          <span class="badge badge-success">IMPROVED</span>
                        {:else}
                          <span class="badge badge-neutral">NEUTRAL</span>
                        {/if}
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<!-- MODAL: CREATE DATASET -->
{#if showCreateDatasetModal}
  <div class="modal-backdrop" onclick={() => (showCreateDatasetModal = false)}>
    <div class="modal-card card" onclick={(e) => e.stopPropagation()}>
      <div class="modal-header">
        <h3>Create Benchmark Dataset</h3>
        <button class="btn-text" onclick={() => (showCreateDatasetModal = false)}>✕</button>
      </div>
      <div class="form-group">
        <label class="form-label">Dataset Name</label>
        <input type="text" class="form-input" bind:value={newDatasetName} placeholder="e.g. Legal Q&A Benchmark" />
      </div>
      <div class="form-group">
        <label class="form-label">Description</label>
        <textarea class="form-textarea" bind:value={newDatasetDesc} placeholder="Describe evaluation objectives..."></textarea>
      </div>
      <div class="modal-footer">
        <button class="btn-outline" onclick={() => (showCreateDatasetModal = false)}>Cancel</button>
        <button class="btn-primary" onclick={handleCreateDataset}>Create Dataset</button>
      </div>
    </div>
  </div>
{/if}

<!-- MODAL: IMPORT JSONL -->
{#if showImportModal}
  <div class="modal-backdrop" onclick={() => (showImportModal = false)}>
    <div class="modal-card card" onclick={(e) => e.stopPropagation()}>
      <div class="modal-header">
        <h3>Import Benchmark Dataset (JSONL)</h3>
        <button class="btn-text" onclick={() => (showImportModal = false)}>✕</button>
      </div>
      <div class="form-group">
        <label class="form-label">Dataset Title</label>
        <input type="text" class="form-input" bind:value={importName} placeholder="Imported Benchmark Title" />
      </div>
      <div class="form-group">
        <label class="form-label">Raw JSONL Content</label>
        <textarea
          class="form-textarea code-area"
          bind:value={importJsonlContent}
          placeholder={`{"query_id": "Q-01", "query": "Example text", "relevance_judgments": [{"chunk_id": "c1", "grade": 2}]}`}
        ></textarea>
      </div>
      <div class="modal-footer">
        <button class="btn-outline" onclick={() => (showImportModal = false)}>Cancel</button>
        <button class="btn-primary" onclick={handleImportDataset}>Validate & Import</button>
      </div>
    </div>
  </div>
{/if}

<!-- MODAL: ADD QUERY -->
{#if showAddQueryModal}
  <div class="modal-backdrop" onclick={() => (showAddQueryModal = false)}>
    <div class="modal-card card" onclick={(e) => e.stopPropagation()}>
      <div class="modal-header">
        <h3>Add Golden Evaluation Query</h3>
        <button class="btn-text" onclick={() => (showAddQueryModal = false)}>✕</button>
      </div>
      <div class="form-group">
        <label class="form-label">Query ID</label>
        <input type="text" class="form-input" bind:value={newQueryId} placeholder="e.g. Q-01" />
      </div>
      <div class="form-group">
        <label class="form-label">Question Text</label>
        <input type="text" class="form-input" bind:value={newQueryText} placeholder="Enter evaluation question..." />
      </div>
      <div class="form-group">
        <label class="form-label">Expected Reference Answer (Optional)</label>
        <textarea class="form-textarea" bind:value={newQueryAnswer} placeholder="Ground truth answer text..."></textarea>
      </div>
      <div class="form-group">
        <label class="form-label">Relevant Chunk IDs (comma-separated)</label>
        <input type="text" class="form-input" bind:value={newQueryChunks} placeholder="doc_1:chunk:0, doc_1:chunk:1" />
      </div>
      <div class="form-group">
        <label class="form-label">Required Citation IDs (comma-separated, optional)</label>
        <input type="text" class="form-input" bind:value={newQueryCitations} placeholder="CIT-01, CIT-02" />
      </div>
      <div class="modal-footer">
        <button class="btn-outline" onclick={() => (showAddQueryModal = false)}>Cancel</button>
        <button class="btn-primary" onclick={handleAddQuery}>Add Query</button>
      </div>
    </div>
  </div>
{/if}

<!-- MODAL: REPRODUCIBILITY -->
{#if showReproducibilityModal && selectedReproducibility}
  <div class="modal-backdrop" onclick={() => (showReproducibilityModal = false)}>
    <div class="modal-card card" onclick={(e) => e.stopPropagation()}>
      <div class="modal-header">
        <h3>Reproducibility Manifest</h3>
        <button class="btn-text" onclick={() => (showReproducibilityModal = false)}>✕</button>
      </div>
      <div class="repro-body">
        <div class="repro-row">
          <span>Can Re-Execute Now:</span>
          <strong>{selectedReproducibility.can_reproduce ? 'YES (Dependencies Verified)' : 'NO (Missing Dependencies)'}</strong>
        </div>
        <div class="repro-row">
          <span>Config Hash:</span>
          <code>{selectedReproducibility.config_hash}</code>
        </div>
        <div class="repro-row">
          <span>Dataset Reference:</span>
          <span>{selectedReproducibility.dataset_id ?? 'None'} ({selectedReproducibility.dataset_exists ? 'Verified' : 'Missing'})</span>
        </div>
        <div class="repro-row">
          <span>Corpus Verified:</span>
          <span>{selectedReproducibility.corpus_documents_exist ? 'All documents available' : 'Some corpus files missing'}</span>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn-primary" onclick={() => (showReproducibilityModal = false)}>Close</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .studio-wrapper {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1.5rem 1.5rem 3rem;
  }

  .studio-header {
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 1.5rem;
  }

  .title-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
  }

  .studio-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
  }

  .version-tag {
    background: #6366f1;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.5rem;
    border-radius: var(--radius-sm);
  }

  .studio-subtitle {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin: 0 0 1.25rem;
  }

  .studio-tabs {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .tab-btn {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    padding: 0.5rem 1rem;
    border-radius: var(--radius-md);
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.15s;
  }

  .tab-btn:hover {
    color: var(--text-primary);
    border-color: var(--border-strong);
  }

  .tab-btn.active {
    background: #4f46e5;
    color: white;
    border-color: #6366f1;
  }

  /* Banners */
  .banner {
    padding: 0.75rem 1.25rem;
    border-radius: var(--radius-md);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 0.875rem;
  }

  .banner-error {
    background: rgba(225, 29, 72, 0.15);
    border: 1px solid rgba(225, 29, 72, 0.4);
    color: #fda4af;
  }

  .banner-success {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #a7f3d0;
  }

  .banner-close {
    margin-left: auto;
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
    font-size: 1rem;
  }

  /* Cards & Grid */
  .card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.75rem;
  }

  .card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .launcher-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }

  @media (max-width: 900px) {
    .launcher-grid {
      grid-template-columns: 1fr;
    }
  }

  /* Forms */
  .form-group {
    margin-bottom: 1.25rem;
  }

  .form-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .col-6 { flex: 1; }
  .col-5 { flex: 5; }
  .col-4 { flex: 4; }
  .col-2 { flex: 2; }

  .form-label {
    display: block;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .form-hint {
    display: block;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
  }

  .form-input, .form-select, .form-textarea {
    width: 100%;
    background: var(--bg-canvas);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    padding: 0.55rem 0.75rem;
    border-radius: var(--radius-sm);
    font-size: 0.875rem;
  }

  .form-input-sm {
    width: 100%;
    background: var(--bg-canvas);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    padding: 0.35rem 0.5rem;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
  }

  .form-range {
    width: 100%;
  }

  .doc-checklist {
    max-height: 180px;
    overflow-y: auto;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    background: var(--bg-canvas);
    padding: 0.5rem;
  }

  .doc-check-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem;
    font-size: 0.85rem;
    cursor: pointer;
    border-radius: var(--radius-sm);
  }

  .doc-check-item:hover {
    background: var(--bg-surface);
  }

  .doc-title { flex: 1; color: var(--text-primary); }
  .doc-meta { font-size: 0.75rem; color: var(--text-muted); }

  .check-actions {
    display: flex;
    gap: 1rem;
    margin-top: 0.4rem;
  }

  .btn-text {
    background: none;
    border: none;
    color: #6366f1;
    font-size: 0.75rem;
    cursor: pointer;
    padding: 0;
  }

  .stages-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }

  .stage-toggle {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--text-primary);
    cursor: pointer;
  }

  .provenance-banner {
    padding: 0.75rem;
    border-radius: var(--radius-md);
    margin-top: 1rem;
    font-size: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .warning-prov {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #fde68a;
  }

  .local-prov {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #a7f3d0;
  }

  .prov-badge {
    font-weight: 700;
    font-size: 0.7rem;
    padding: 0.2rem 0.4rem;
    border-radius: var(--radius-sm);
    background: rgba(0, 0, 0, 0.3);
  }

  .launch-footer {
    margin-top: 1.5rem;
    border-top: 1px solid var(--border-subtle);
    padding-top: 1.25rem;
  }

  .launch-btn {
    width: 100%;
    padding: 0.85rem;
    font-size: 1rem;
    font-weight: 600;
    border-radius: var(--radius-md);
    cursor: pointer;
  }

  .btn-primary {
    background: #4f46e5;
    color: white;
    border: none;
  }

  .btn-outline {
    background: transparent;
    border: 1px solid var(--border-strong);
    color: var(--text-primary);
    cursor: pointer;
    border-radius: var(--radius-sm);
    padding: 0.4rem 0.75rem;
  }

  .btn-sm {
    font-size: 0.8rem;
    padding: 0.35rem 0.65rem;
  }

  /* Progress Box */
  .job-progress-box {
    margin-bottom: 1rem;
    background: var(--bg-canvas);
    padding: 0.75rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-subtle);
  }

  .progress-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    margin-bottom: 0.4rem;
  }

  .progress-track {
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 9999px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: #4f46e5;
    transition: width 0.3s;
  }

  /* Datasets Layout */
  .dataset-layout {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 1.5rem;
  }

  .dataset-sidebar {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1rem;
  }

  .sidebar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.5rem;
  }

  .sidebar-title {
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0;
  }

  .dataset-list {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .dataset-item-btn {
    text-align: left;
    background: var(--bg-canvas);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
    color: var(--text-primary);
  }

  .dataset-item-btn:hover {
    border-color: #6366f1;
  }

  .dataset-item-btn.selected {
    border-color: #6366f1;
    background: rgba(99, 102, 241, 0.1);
  }

  .ds-item-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
  }

  .ds-item-name { font-weight: 600; font-size: 0.85rem; }
  .ds-item-meta { font-size: 0.75rem; color: var(--text-muted); display: flex; gap: 0.75rem; }

  .dh-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .dh-title { font-size: 1.25rem; font-weight: 700; margin: 0 0 0.35rem; }
  .dh-desc { font-size: 0.85rem; color: var(--text-secondary); margin: 0; }
  .dh-tags { display: flex; gap: 0.75rem; margin-top: 1rem; font-size: 0.8rem; }
  .tag { background: var(--bg-canvas); padding: 0.2rem 0.5rem; border-radius: var(--radius-sm); }

  /* Tables */
  .table-container {
    overflow-x: auto;
  }

  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }

  .data-table th, .data-table td {
    padding: 0.65rem 0.75rem;
    border-bottom: 1px solid var(--border-subtle);
    text-align: left;
  }

  .data-table th {
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
  }

  .chunks-pill-group {
    display: flex;
    gap: 0.35rem;
    flex-wrap: wrap;
  }

  .chunk-pill {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    font-size: 0.75rem;
    padding: 0.15rem 0.4rem;
    border-radius: var(--radius-sm);
  }

  .btn-danger-sm {
    background: rgba(225, 29, 72, 0.2);
    color: #fda4af;
    border: none;
    border-radius: var(--radius-sm);
    padding: 0.25rem 0.5rem;
    cursor: pointer;
  }

  /* Runs Grid */
  .runs-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 1.25rem;
  }

  .run-card {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .is-baseline-card {
    border-color: #f59e0b;
  }

  .rc-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.75rem;
  }

  .rc-id { font-size: 0.75rem; color: var(--text-muted); }
  .rc-name { font-size: 1rem; font-weight: 600; margin: 0.2rem 0 0; }

  .rc-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
    margin: 1rem 0;
    background: var(--bg-canvas);
    padding: 0.5rem;
    border-radius: var(--radius-sm);
  }

  .metric-pill {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .mp-label { font-size: 0.7rem; color: var(--text-muted); }
  .mp-val { font-size: 0.9rem; font-weight: 700; color: var(--text-primary); }

  .rc-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid var(--border-subtle);
    padding-top: 0.75rem;
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  /* Failure Explorer */
  .failures-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .failure-card {
    cursor: pointer;
    transition: border-color 0.15s;
  }

  .failure-card.has-defect {
    border-left: 4px solid #ef4444;
  }

  .fc-header {
    display: flex;
    justify-content: space-between;
  }

  .fc-id-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.35rem;
  }

  .fc-query-text {
    font-size: 1rem;
    margin: 0.25rem 0;
    color: var(--text-primary);
  }

  .fc-explanation {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 0;
  }

  .fc-expanded-body {
    margin-top: 1rem;
    border-top: 1px solid var(--border-subtle);
    padding-top: 1rem;
  }

  .fc-side-by-side {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }

  .f-subcard {
    background: var(--bg-canvas);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem;
  }

  .sub-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0 0 0.5rem;
  }

  .cand-item {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.5rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
  }

  .cand-item.is-relevant {
    border-color: #10b981;
  }

  /* Regression Studio */
  .verdict-banner {
    padding: 1.25rem;
    border-radius: var(--radius-md);
    margin-bottom: 1.5rem;
  }

  .v-pass {
    background: rgba(16, 185, 129, 0.15);
    border: 2px solid #10b981;
    color: #a7f3d0;
  }

  .v-regression {
    background: rgba(239, 68, 68, 0.15);
    border: 2px solid #ef4444;
    color: #fca5a5;
  }

  .v-incomplete {
    background: rgba(245, 158, 11, 0.15);
    border: 2px solid #f59e0b;
    color: #fde68a;
  }

  .verdict-label {
    font-size: 1.25rem;
    margin-bottom: 0.4rem;
  }

  /* Badges */
  .badge {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: var(--radius-sm);
  }

  .badge-synthetic { background: #8b5cf6; color: white; }
  .badge-baseline { background: #f59e0b; color: black; }
  .badge-prov-mock { background: #6b7280; color: white; }
  .badge-prov-local { background: #10b981; color: white; }
  .badge-prov-cloud { background: #3b82f6; color: white; }
  .badge-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; }
  .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
  .badge-success { background: rgba(16, 185, 129, 0.2); color: #34d399; }
  .badge-neutral { background: rgba(255, 255, 255, 0.1); color: var(--text-muted); }

  /* Modal */
  .modal-backdrop {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0, 0, 0, 0.75);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .modal-card {
    width: 90%;
    max-width: 600px;
    max-height: 90vh;
    overflow-y: auto;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.25rem;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.75rem;
    margin-top: 1.5rem;
  }
</style>

