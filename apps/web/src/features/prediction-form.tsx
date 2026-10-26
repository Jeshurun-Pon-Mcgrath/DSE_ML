import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Button, Notice } from '@/components/ui';
import { predictionSchema, request } from '@/lib/api';
import { useModelInfo } from '@/lib/queries';
const id = z
  .string()
  .min(1)
  .max(100)
  .regex(/^[A-Za-z0-9_-]+$/, 'Use letters, numbers, underscores or hyphens');
const schema = z.strictObject({
  TransactionID: id,
  TransactionDT: z.number().finite().min(0).max(1e12),
  TransactionAmt: z.number().finite().min(0).max(1e12),
  behavioural_entity_id: id,
  merchant_id: z.string().max(100),
  device_id: z.string().max(100),
  billing_region: z.string().max(80),
  merchant_category: z.string().max(80).optional(),
  product_code: z.string().max(80).optional(),
  failed_attempts_before: z.number().finite().int().min(0).max(100000).nullable().optional(),
  account_age_days: z.number().finite().min(0).max(100000).nullable().optional(),
  is_international: z.number().int().min(0).max(1).nullable().optional(),
});
type Input = z.infer<typeof schema>;
export default function PredictionForm({ onClose }: { onClose: () => void }) {
  const model = useModelInfo();
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [batch, setBatch] = useState('');
  const [batchError, setBatchError] = useState('');
  const navigate = useNavigate();
  const client = useQueryClient();
  const form = useForm<Input>({
    resolver: zodResolver(schema),
    defaultValues: {
      TransactionID: '',
      behavioural_entity_id: '',
      merchant_id: '',
      device_id: '',
      billing_region: '',
    },
  });
  const mutation = useMutation({
    mutationFn: (items: Record<string, unknown>[]) =>
      items.length === 1
        ? request('/predict', predictionSchema, {
            method: 'POST',
            body: JSON.stringify(items[0]),
          }).then((p) => [p])
        : request('/predict/batch', z.array(predictionSchema), {
            method: 'POST',
            body: JSON.stringify({ transactions: items }),
          }),
    onSuccess: async (predictions) => {
      await Promise.all(
        ['transactions', 'dashboard', 'audit', 'model-info'].map((key) =>
          client.invalidateQueries({ queryKey: [key] }),
        ),
      );
      onClose();
      const first = predictions[0];
      if (first) await navigate(`/app/transactions/${first.transaction_id}`);
    },
  });
  const submit = form.handleSubmit((v) => {
    const tx = Object.fromEntries(
      Object.entries(v).map(([key, value]) => [key, value === '' ? null : value]),
    );
    mutation.mutate([tx]);
  });
  function submitBatch() {
    try {
      const raw: unknown = JSON.parse(batch);
      const checked = z.array(z.record(z.string(), z.json())).min(1).max(500).parse(raw);
      setBatchError('');
      mutation.mutate(checked);
    } catch {
      setBatchError(
        'Enter a JSON array with 1–500 transaction objects. The API validates every field.',
      );
    }
  }
  return (
    <div className="page-stack">
      <div className="filters">
        <Button className={mode === 'single' ? '' : 'secondary'} onClick={() => setMode('single')}>
          Single transaction
        </Button>
        <Button className={mode === 'batch' ? '' : 'secondary'} onClick={() => setMode('batch')}>
          JSON batch
        </Button>
      </div>
      <Notice>
        Use your own synthetic transaction inputs. No ground-truth labels are accepted. Elapsed time
        must be at least {model.data?.minimum_elapsed_seconds ?? 'the current system watermark'}{' '}
        seconds.
      </Notice>
      {mode === 'single' ? (
        <form
          onSubmit={(event) => {
            void submit(event);
          }}
        >
          <div className="form-grid">
            {(
              [
                ['TransactionID', 'Transaction ID', 'text'],
                ['behavioural_entity_id', 'Behavioural entity ID', 'text'],
                ['TransactionAmt', 'Amount (INR)', 'number'],
                ['TransactionDT', 'Elapsed seconds', 'number'],
                ['merchant_id', 'Merchant ID (optional)', 'text'],
                ['device_id', 'Device ID (optional)', 'text'],
                ['billing_region', 'Billing region (optional)', 'text'],
              ] as const
            ).map(([key, label, type]) => (
              <div className="field" key={key}>
                <label htmlFor={key}>{label}</label>
                <input
                  id={key}
                  type={type}
                  step={type === 'number' ? 'any' : undefined}
                  {...form.register(key, type === 'number' ? { valueAsNumber: true } : {})}
                  aria-invalid={!!form.formState.errors[key]}
                  aria-describedby={form.formState.errors[key] ? `${key}-error` : undefined}
                />
                {form.formState.errors[key] && (
                  <span id={`${key}-error`} className="field-error">
                    {form.formState.errors[key]?.message}
                  </span>
                )}
              </div>
            ))}
          </div>
          <details className="research-card">
            <summary>Additional transaction context</summary>
            <div className="form-grid">
              {(
                [
                  ['merchant_category', 'Merchant category', 'text'],
                  ['product_code', 'Product code', 'text'],
                  ['failed_attempts_before', 'Failed attempts before', 'number'],
                  ['account_age_days', 'Account age in days', 'number'],
                ] as const
              ).map(([key, label, type]) => (
                <div className="field" key={key}>
                  <label htmlFor={key}>{label} (optional)</label>
                  <input
                    id={key}
                    type={type}
                    step={type === 'number' ? 'any' : undefined}
                    {...form.register(
                      key,
                      type === 'number'
                        ? {
                            setValueAs: (value: unknown) =>
                              value == null || value === ''
                                ? null
                                : typeof value === 'string' || typeof value === 'number'
                                  ? Number(value)
                                  : null,
                          }
                        : {},
                    )}
                    aria-invalid={!!form.formState.errors[key]}
                  />
                  {form.formState.errors[key] && (
                    <span role="alert" className="field-error">
                      {form.formState.errors[key]?.message}
                    </span>
                  )}
                </div>
              ))}
              <div className="field">
                <label htmlFor="is_international">International context (optional)</label>
                <select
                  id="is_international"
                  {...form.register('is_international', {
                    setValueAs: (value: unknown) =>
                      value === '' || value == null ? null : Number(value),
                  })}
                >
                  <option value="">Unknown</option>
                  <option value="0">Domestic</option>
                  <option value="1">International</option>
                </select>
              </div>
            </div>
          </details>
          <div className="form-actions">
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? 'Computing evidence…' : 'Generate prediction'}
            </Button>
          </div>
        </form>
      ) : (
        <div className="field">
          <label htmlFor="batch-json">Transaction objects (JSON array)</label>
          <textarea
            id="batch-json"
            rows={10}
            value={batch}
            onChange={(e) => setBatch(e.target.value)}
            spellCheck={false}
          />
          {batchError && (
            <span role="alert" className="field-error">
              {batchError}
            </span>
          )}
          <Button disabled={mutation.isPending} onClick={submitBatch}>
            Submit bounded batch
          </Button>
        </div>
      )}
      {mutation.isError && (
        <p className="field-error" role="alert">
          {mutation.error.message}
        </p>
      )}
      <p className="caption">
        Submitting saves the prediction and audit event. Duplicate IDs and out-of-order events are
        rejected. In production, writes require the configured trusted operator proxy.
      </p>
    </div>
  );
}
