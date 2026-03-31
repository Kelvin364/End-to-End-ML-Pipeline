"use client";

import { useState, useRef, DragEvent } from "react";
import { api, StatusResponse } from "@/lib/api";
import { SectionHeader } from "@/components/SectionHeader";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { ErrorMessage } from "@/components/ErrorMessage";
import { ProgressBar } from "@/components/ProgressBar";

const CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"];

export default function UploadPage() {
  const [label,     setLabel]     = useState("EOSINOPHIL");
  const [files,     setFiles]     = useState<File[]>([]);
  const [dragging,  setDragging]  = useState(false);
  const [uploading, setUploading] = useState(false);
  const [retraining,setRetraining]= useState(false);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [retainMsg, setRetainMsg] = useState<string | null>(null);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [retainErr, setRetainErr] = useState<string | null>(null);
  const [pending,   setPending]   = useState<number | null>(null);
  const [threshold, setThreshold] = useState(50);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (newFiles: FileList | File[]) => {
    const valid = Array.from(newFiles).filter(f => f.type.startsWith("image/"));
    setFiles(prev => [...prev, ...valid]);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const removeFile = (i: number) =>
    setFiles(prev => prev.filter((_, j) => j !== i));

  const runUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    setUploadMsg(null);
    setUploadErr(null);
    try {
      const res = await api.upload(files, label);
      setUploadMsg(
        `${res.uploaded} image${res.uploaded !== 1 ? "s" : ""} uploaded. ` +
        `Failed: ${res.failed}. ` +
        `Pending for retraining: ${res.pending_total} / ${res.retrain_threshold}.` +
        (res.auto_retrain_triggered
          ? " Threshold reached — retraining triggered automatically."
          : "")
      );
      setPending(res.pending_total);
      setThreshold(res.retrain_threshold);
      setFiles([]);
    } catch (e: unknown) {
      setUploadErr(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const runRetrain = async () => {
    setRetraining(true);
    setRetainMsg(null);
    setRetainErr(null);
    try {
      const res = await api.retrain();
      if (res.status === "skipped") {
        setRetainMsg(`Skipped: ${res.reason}`);
      } else {
        setRetainMsg(
          res.message ||
          `Retraining triggered on ${res.pending} images via ${res.runner}. ` +
          `RAM available: ${res.ram_available}.`
        );
      }
    } catch (e: unknown) {
      setRetainErr(e instanceof Error ? e.message : "Retrain failed");
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-foreground tracking-tight">
          Upload & Retrain
        </h1>
        <p className="mt-1 text-sm text-secondary">
          Upload labelled cell images for model retraining.
          Retraining executes autonomously on GitHub Actions (7 GB RAM).
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload section */}
        <div>
          <SectionHeader
            title="Bulk Image Upload"
            description="All images in a batch receive the same label. Select the correct cell type before uploading."
          />

          {/* Label selector */}
          <div className="mb-4">
            <label className="block text-xs font-medium uppercase tracking-widest
              text-secondary mb-2">
              Cell Type Label
            </label>
            <div className="grid grid-cols-2 gap-2">
              {CLASSES.map(cls => (
                <button
                  key={cls}
                  onClick={() => setLabel(cls)}
                  className={`
                    py-2 px-3 rounded-md text-xs font-mono text-left
                    border transition-all duration-150
                    ${label === cls
                      ? "border-accent bg-accent/10 text-foreground"
                      : "border-border text-secondary hover:border-muted hover:text-foreground"
                    }
                  `}
                >
                  {cls}
                </button>
              ))}
            </div>
          </div>

          {/* Drop zone */}
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`
              rounded-lg border-2 border-dashed cursor-pointer
              transition-all duration-200 p-8 text-center
              ${dragging
                ? "border-accent bg-accent/5"
                : "border-border hover:border-muted bg-surface"
              }
            `}
          >
            <p className="text-sm text-secondary">
              Drag and drop images or click to browse
            </p>
            <p className="text-xs text-secondary/60 mt-1">
              Multiple files supported — JPEG or PNG
            </p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            multiple
            className="hidden"
            onChange={e => e.target.files && handleFiles(e.target.files)}
          />

          {/* File list */}
          {files.length > 0 && (
            <div className="mt-3 rounded-lg border border-border bg-surface
              divide-y divide-border max-h-40 overflow-y-auto">
              {files.map((f, i) => (
                <div key={i}
                  className="flex items-center justify-between px-4 py-2.5">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-xs font-mono text-secondary shrink-0">
                      {(f.size / 1024).toFixed(0)}KB
                    </span>
                    <span className="text-xs text-foreground truncate">
                      {f.name}
                    </span>
                  </div>
                  <button
                    onClick={e => { e.stopPropagation(); removeFile(i); }}
                    className="text-secondary hover:text-danger text-xs ml-3 shrink-0
                      transition-colors"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={runUpload}
            disabled={!files.length || uploading}
            className="
              mt-4 w-full py-2.5 rounded-md text-sm font-medium
              bg-accent text-white hover:bg-accent-hover
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-all duration-150 flex items-center justify-center gap-2
            "
          >
            {uploading ? (
              <><LoadingSpinner size={14} /> Uploading {files.length} file{files.length !== 1 ? "s" : ""}...</>
            ) : (
              `Upload ${files.length > 0 ? `${files.length} ` : ""}Image${files.length !== 1 ? "s" : ""}`
            )}
          </button>

          {uploadMsg && (
            <div className="mt-3 rounded-lg border border-success/20
              bg-success/5 p-4">
              <p className="text-xs text-success font-mono">{uploadMsg}</p>
            </div>
          )}
          {uploadErr && (
            <div className="mt-3"><ErrorMessage message={uploadErr} /></div>
          )}

          {pending !== null && (
            <div className="mt-4">
              <ProgressBar
                value={pending}
                max={threshold}
                label="Pending images for auto-retrain"
              />
            </div>
          )}
        </div>

        {/* Retrain section */}
        <div>
          <SectionHeader
            title="Trigger Retraining"
            description="Initiates the autonomous retraining workflow on GitHub Actions.
              The job has 7 GB RAM and runs TensorFlow fine-tuning for up to 5 epochs.
              The model is saved back to Supabase Storage only if weighted F1 improves."
          />

          <div className="rounded-lg border border-border bg-surface p-5 mb-5">
            <p className="text-xs font-medium uppercase tracking-widest
              text-secondary mb-4">
              Retraining Architecture
            </p>
            <div className="space-y-3">
              {[
                ["Compute",   "GitHub Actions — ubuntu-latest"],
                ["RAM",       "7 GB (free tier)"],
                ["Framework", "TensorFlow 2.19 + Keras 3.13"],
                ["Strategy",  "Fine-tune on new images, 5 epochs max"],
                ["Guard",     "Save only if weighted F1 improves"],
                ["Storage",   "Model saved to Supabase Storage"],
                ["Logging",   "Run logged to retraining_runs table"],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-xs text-secondary">{k}</span>
                  <span className="text-xs font-mono text-foreground">{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-border bg-surface p-5 mb-5">
            <p className="text-xs font-medium uppercase tracking-widest
              text-secondary mb-4">
              Autonomous Trigger Logic
            </p>
            <div className="space-y-3 text-xs font-mono text-secondary">
              <div className="flex items-start gap-2">
                <span className="text-accent shrink-0">01</span>
                <span>User uploads images via this page</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-accent shrink-0">02</span>
                <span>Images stored in Supabase Storage, flagged retrained=false</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-accent shrink-0">03</span>
                <span>Pending count checked against threshold (50 images)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-accent shrink-0">04</span>
                <span>Threshold reached — GitHub Actions dispatch event fired</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-accent shrink-0">05</span>
                <span>Retraining job executes, result logged to database</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-accent shrink-0">06</span>
                <span>History tab updated with run outcome</span>
              </div>
            </div>
          </div>

          <button
            onClick={runRetrain}
            disabled={retraining}
            className="
              w-full py-2.5 rounded-md text-sm font-medium
              border border-foreground/20 text-foreground
              hover:bg-foreground/5
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-all duration-150 flex items-center justify-center gap-2
            "
          >
            {retraining ? (
              <><LoadingSpinner size={14} /> Triggering...</>
            ) : (
              "Trigger Retraining Now"
            )}
          </button>

          {retainMsg && (
            <div className="mt-3 rounded-lg border border-accent/20
              bg-accent/5 p-4">
              <p className="text-xs text-foreground font-mono">{retainMsg}</p>
            </div>
          )}
          {retainErr && (
            <div className="mt-3"><ErrorMessage message={retainErr} /></div>
          )}
        </div>
      </div>
    </div>
  );
}
