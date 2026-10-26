import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { z } from 'zod';
import {
  auditSchema,
  dashboardSchema,
  explanationSchema,
  finalSchema,
  hardSchema,
  healthSchema,
  modelSchema,
  request,
  scenariosSchema,
  transactionPageSchema,
  transactionSchema,
  type ReviewState,
} from './api';
export const useDashboardSummary = () =>
  useQuery({
    queryKey: ['dashboard'],
    queryFn: ({ signal }) => request('/dashboard/summary', dashboardSchema, {}, signal),
  });
export const useFinalEvaluation = () =>
  useQuery({
    queryKey: ['final'],
    queryFn: ({ signal }) => request('/evaluation/final', finalSchema, {}, signal),
    staleTime: Infinity,
  });
export const useScenarios = () =>
  useQuery({
    queryKey: ['scenarios'],
    queryFn: ({ signal }) => request('/evaluation/scenarios', scenariosSchema, {}, signal),
    staleTime: Infinity,
  });
export const useHardNegatives = () =>
  useQuery({
    queryKey: ['hard-negatives'],
    queryFn: ({ signal }) => request('/evaluation/hard-negatives', hardSchema, {}, signal),
    staleTime: Infinity,
  });
export const useModelInfo = () =>
  useQuery({
    queryKey: ['model-info'],
    queryFn: ({ signal }) => request('/system/model', modelSchema, {}, signal),
  });
export const useHealth = () =>
  useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => request('/system/health', healthSchema, {}, signal),
    refetchInterval: 60000,
  });
export const useTransactions = (params: Record<string, string>) =>
  useQuery({
    queryKey: ['transactions', params],
    queryFn: ({ signal }) =>
      request(`/transactions?${new URLSearchParams(params)}`, transactionPageSchema, {}, signal),
    placeholderData: keepPreviousData,
  });
export const useTransaction = (id: string) =>
  useQuery({
    queryKey: ['transaction', id],
    queryFn: ({ signal }) =>
      request(`/transactions/${encodeURIComponent(id)}`, transactionSchema, {}, signal),
    enabled: !!id,
  });
export const useExplanation = (id: string) =>
  useQuery({
    queryKey: ['explanation', id],
    queryFn: ({ signal }) =>
      request(`/transactions/${encodeURIComponent(id)}/explanation`, explanationSchema, {}, signal),
    enabled: !!id,
    staleTime: Infinity,
  });
export const useReport = (path: string) =>
  useQuery({
    queryKey: ['report', path],
    queryFn: ({ signal }) => request(path, z.json(), {}, signal),
    staleTime: 300000,
  });
export const useAudit = (page: number, id = '') =>
  useQuery({
    queryKey: ['audit', page, id],
    queryFn: ({ signal }) =>
      request(
        `/audit?${new URLSearchParams({ page: String(page), page_size: '25', ...(id ? { transaction_id: id } : {}) })}`,
        auditSchema,
        {},
        signal,
      ),
    placeholderData: keepPreviousData,
  });
export function useInvestigation(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (state: ReviewState) =>
      request(
        `/transactions/${encodeURIComponent(id)}/investigation`,
        z.object({
          transaction_id: z.string(),
          state: z.string(),
          scores_unchanged: z.literal(true),
        }),
        { method: 'PATCH', body: JSON.stringify({ state }) },
      ),
    onSuccess: async () => {
      await Promise.all(
        ['transactions', 'transaction', 'dashboard', 'audit'].map((key) =>
          client.invalidateQueries({ queryKey: [key] }),
        ),
      );
    },
  });
}
