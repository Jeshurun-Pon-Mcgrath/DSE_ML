import { ArrowUpRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { DataTable, RiskBadge } from '@/components/ui';
import type { Transaction } from '@/lib/api';
import { amount, elapsed, fmt, human, pct } from '@/lib/format';
export function TransactionTable({
  items,
  details = true,
}: {
  items: Transaction[];
  details?: boolean;
}) {
  return (
    <DataTable
      label="Transactions"
      headers={
        details
          ? [
              'Transaction',
              'Amount',
              'Elapsed time',
              'Risk',
              'Review state',
              'Supervised P',
              'Behaviour B',
            ]
          : ['Transaction', 'Amount', 'Fused risk', 'Risk level']
      }
      rows={items.map((t) => ({
        id: t.prediction.transaction_id,
        href: `/app/transactions/${t.prediction.transaction_id}`,
        cells: details
          ? [
              <Link className="row-link" to={`/app/transactions/${t.prediction.transaction_id}`}>
                {t.prediction.transaction_id}
                <ArrowUpRight size={13} />
              </Link>,
              amount(
                typeof t.transaction.TransactionAmt === 'number'
                  ? t.transaction.TransactionAmt
                  : null,
              ),
              typeof t.transaction.TransactionDT === 'number'
                ? elapsed(t.transaction.TransactionDT)
                : 'Unavailable',
              <div className="small-gap">
                <span className="tabular">{fmt(t.prediction.risk.score, 4)}</span>
                <RiskBadge level={t.prediction.risk.level} />
              </div>,
              human(t.investigation),
              pct(t.prediction.supervised.probability),
              fmt(t.prediction.behaviour.score, 3),
            ]
          : [
              <Link className="row-link" to={`/app/transactions/${t.prediction.transaction_id}`}>
                {t.prediction.transaction_id}
                <ArrowUpRight size={13} />
              </Link>,
              amount(
                typeof t.transaction.TransactionAmt === 'number'
                  ? t.transaction.TransactionAmt
                  : null,
              ),
              fmt(t.prediction.risk.score, 4),
              <RiskBadge level={t.prediction.risk.level} />,
            ],
      }))}
    />
  );
}
