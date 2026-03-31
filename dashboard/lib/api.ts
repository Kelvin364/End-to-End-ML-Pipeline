const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://blood-cell-api.onrender.com";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { ...options?.headers },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} failed: ${res.status} — ${text.slice(0, 200)}`);
  }
  return res.json();
}

export interface StatusResponse {
  status            : string;
  uptime_seconds    : number;
  uptime_human      : string;
  model_name        : string;
  accuracy          : number | null;
  f1                : number | null;
  model_exists      : boolean;
  scheduler_running : boolean;
  pending_images_for_retrain: number;
  retrain_threshold : number;
  next_auto_retrain : string;
}

export interface MetricsResponse {
  per_class: Record<string, {
    precision : number;
    recall    : number;
    f1        : number;
    support   : number;
  }>;
  overall: {
    accuracy           : number;
    weighted_f1        : number;
    weighted_precision : number;
    weighted_recall    : number;
    total_samples      : number;
    num_classes        : number;
  };
  models_comparison: Array<{
    name      : string;
    accuracy  : number;
    f1        : number;
    precision : number;
    recall    : number;
    params    : number;
    epochs    : number;
    best      : boolean;
  }>;
  class_distribution: Record<string, number>;
  dataset_info: {
    total_images : number;
    train_split  : number;
    val_split    : number;
    test_split   : number;
    image_size   : string;
    stain        : string;
  };
}

export interface PredictionResponse {
  filename   : string;
  prediction : {
    label      : string;
    confidence : number;
    all_scores : Record<string, number>;
  };
  status: string;
}

export interface UploadResponse {
  uploaded               : number;
  failed                 : number;
  results                : Array<{ filename: string; status: string }>;
  pending_total          : number;
  auto_retrain_triggered : boolean;
  retrain_threshold      : number;
}

export interface RetrainResponse {
  status    : string;
  message   : string;
  pending   : number;
  runner    : string;
  ram_available: string;
  reason    ?: string;
}

export interface HistoryRun {
  id           : string;
  triggered_at : string;
  triggered_by : string;
  images_used  : number | null;
  f1_before    : number | null;
  f1_after     : number | null;
  improved     : boolean | null;
  duration_s   : number | null;
  epochs_run   : number | null;
}

export const api = {
  status   : ()      => apiFetch<StatusResponse>("/status"),
  metrics  : ()      => apiFetch<MetricsResponse>("/metrics"),
  history  : ()      => apiFetch<{ runs: HistoryRun[] }>("/history"),
  health   : ()      => apiFetch<{ status: string; uptime_s: number }>("/"),

  predict: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return apiFetch<PredictionResponse>("/predict", { method: "POST", body: fd });
  },

  upload: (files: File[], label: string) => {
    const fd = new FormData();
    files.forEach(f => fd.append("files", f));
    return apiFetch<UploadResponse>(`/upload?label=${label}`, {
      method: "POST", body: fd
    });
  },

  retrain: () => apiFetch<RetrainResponse>("/retrain", { method: "POST" }),
};
