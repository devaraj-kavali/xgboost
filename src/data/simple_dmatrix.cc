/*!
 * Copyright 2014 by Contributors
 * \file simple_dmatrix.cc
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen
 */
#include "./simple_dmatrix.h"
#include <xgboost/data.h>
#include "./simple_batch_iterator.h"
#include "../common/random.h"

namespace xgboost {
namespace data {
MetaInfo& SimpleDMatrix::Info() { return source_->info; }

const MetaInfo& SimpleDMatrix::Info() const { return source_->info; }

float SimpleDMatrix::GetColDensity(size_t cidx) {
  size_t column_size = 0;
  // Use whatever version of column batches already exists
  if (sorted_column_page_) {
    auto batch = this->GetBatches<SortedCSCPage>();
    column_size = (*batch.begin())[cidx].size();
  } else {
    auto batch = this->GetBatches<CSCPage>();
    column_size = (*batch.begin())[cidx].size();
  }

  size_t nmiss = this->Info().num_row_ - column_size;
  return 1.0f - (static_cast<float>(nmiss)) / this->Info().num_row_;
}

BatchSet<SparsePage> SimpleDMatrix::GetRowBatches() {
  // since csr is the default data structure so `source_` is always available.
  auto cast = dynamic_cast<SimpleCSRSource*>(source_.get());
  auto begin_iter = BatchIterator<SparsePage>(
      new SimpleBatchIteratorImpl<SparsePage>(&(cast->page_)));
  return BatchSet<SparsePage>(begin_iter);
}

BatchSet<CSCPage> SimpleDMatrix::GetColumnBatches() {
  // column page doesn't exist, generate it
  if (!column_page_) {
    auto page = dynamic_cast<SimpleCSRSource*>(source_.get())->page_;
    column_page_.reset(new CSCPage(page.GetTranspose(source_->info.num_col_)));
  }
  auto begin_iter =
      BatchIterator<CSCPage>(new SimpleBatchIteratorImpl<CSCPage>(column_page_.get()));
  return BatchSet<CSCPage>(begin_iter);
}

BatchSet<SortedCSCPage> SimpleDMatrix::GetSortedColumnBatches() {
  // Sorted column page doesn't exist, generate it
  if (!sorted_column_page_) {
    auto page = dynamic_cast<SimpleCSRSource*>(source_.get())->page_;
    sorted_column_page_.reset(
        new SortedCSCPage(page.GetTranspose(source_->info.num_col_)));
    sorted_column_page_->SortRows();
  }
  auto begin_iter = BatchIterator<SortedCSCPage>(
      new SimpleBatchIteratorImpl<SortedCSCPage>(sorted_column_page_.get()));
  return BatchSet<SortedCSCPage>(begin_iter);
}

BatchSet<EllpackPage> SimpleDMatrix::GetEllpackBatches() {
  // ELLPACK page doesn't exist, generate it
  if (!ellpack_page_) {
    ellpack_page_.reset(new EllpackPage(this));
  }
  auto begin_iter =
      BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_page_.get()));
  return BatchSet<EllpackPage>(begin_iter);
}

bool SimpleDMatrix::SingleColBlock() const { return true; }

// static members
BatchedDMatrix* BatchedDMatrix::newMat_{nullptr};
std::mutex BatchedDMatrix::batchMutex_;
std::condition_variable BatchedDMatrix::batchBuiltCondVar_;
BatchedDMatrix* BatchedDMatrix::getBatchedDMatrix(int numBatches) {
  std::lock_guard<std::mutex> lg(batchMutex_);
  if (!newMat_) {
    newMat_ = new BatchedDMatrix(numBatches);
  }
  return newMat_;
}

void BatchedDMatrix::CreateInfo() {
  info_->Clear();
  for (const auto& src : sources_) {
    // labels
    auto& src_labels = src->info.labels_.HostVector();
    auto& labels = info_->labels_.HostVector();
    labels.insert(labels.end(), src_labels.begin(), src_labels.end());
    // weights
    auto& src_weights = src->info.weights_.HostVector();
    auto& weights = info_->weights_.HostVector();
    weights.insert(weights.end(), src_weights.begin(), src_weights.end());
    // group_ptr
    auto& src_gptr = src->info.group_ptr_;
    auto& gptr = info_->group_ptr_;
    gptr.insert(gptr.end(), src_gptr.begin(), src_gptr.end());
    // num_row
    info_->num_row_ += src->info.num_row_;
    // num_col
    if (info_->num_col_ == 0) {
      info_->num_col_ = src->info.num_col_;
    } else {
      CHECK_EQ(info_->num_col_, src->info.num_col_) << "invalid data, num_col mismatch";
    }
    // num_nonzero
    info_->num_nonzero_ += src->info.num_nonzero_;
  }
}

MetaInfo& BatchedDMatrix::Info() { 
  return *info_;
}

const MetaInfo& BatchedDMatrix::Info() const { 
  return *info_;
}

bool BatchedDMatrix::AddBatch(std::unique_ptr<SimpleCSRSource>&& batch) {
  std::unique_lock<std::mutex> ul(batchMutex_);
  sources_.push_front(std::move(batch));
  if (++nSources_ == nBatches_) {
    CreateInfo();
    newMat_ = nullptr;
    // notify all
    batchBuiltCondVar_.notify_all();
    return true;
  } else {
    // this thread doesn't continue until notified
    batchBuiltCondVar_.wait(ul);
    return false;
  }
}

float BatchedDMatrix::GetColDensity(size_t cidx) {
  // for now, assuming BatchedDMatrix is always dense
  return 1.0f;
}

BatchSet<SparsePage> BatchedDMatrix::GetRowBatches() {
  std::lock_guard<std::mutex> lg(batchMutex_);
  auto begin_iter = BatchIterator<SparsePage>(new BatchSetIteratorImpl(sources_));
  return BatchSet<SparsePage>(begin_iter);
}

BatchSet<CSCPage> BatchedDMatrix::GetColumnBatches() {
  LOG(FATAL) << "method not implemented";
}

BatchSet<SortedCSCPage> BatchedDMatrix::GetSortedColumnBatches() {
  LOG(FATAL) << "method not implemented";
}

BatchSet<EllpackPage> BatchedDMatrix::GetEllpackBatches() {
  LOG(FATAL) << "method not implemented";
}

bool BatchedDMatrix::SingleColBlock() const { return true; }

size_t BatchedDMatrix::GetNumRows() {
  size_t nrows = 0;
  auto batches = GetRowBatches();
  for (const auto& batch : batches) {
    // std::cout << "batch size = " << batch.Size() << "\n";
    nrows += batch.Size();
  }
  return nrows;
}

}  // namespace data
}  // namespace xgboost
