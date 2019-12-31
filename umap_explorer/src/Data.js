import React, { Component } from 'react'
import Layout from './Layout'
import * as _ from 'lodash'
import * as d3 from 'd3'

let algorithm_options = ['UMAP 5d (0 vs 2)', 'UMAP 5d (1 vs 2)', 'UMAP 5d (0 vs 1)']
let algorithm_embedding_keys = [
  'umap_dims_0_2',
  'umap_dims_1_2',
  'umap_dims_0_1',
]

class Data extends Component {
  constructor(props) {
    super(props)
    this.state = {
      umap_dims_0_2: null,
      mnist_labels: null,
      mnist_dates: null,
      umap_dims_0_1: null,
      umap_dims_1_2: null
    }
  }

  scaleEmbeddings(embeddings) {
    let xs = embeddings.map(e => Math.abs(e[0]))
    let ys = embeddings.map(e => Math.abs(e[1]))
    let max_x = _.max(xs)
    let max_y = _.max(ys)
    let max = Math.max(max_x, max_y)
    let scale = d3
      .scaleLinear()
      .domain([-max, max])
      .range([-20, 20])
    let scaled_embeddings = embeddings.map(e => [scale(e[0]), scale(e[1])])
    return scaled_embeddings
  }

  componentDidMount() {
    fetch(`${process.env.PUBLIC_URL}/umap_dims_0_2.json`)
      .then(response => response.json())
      .then(umap_embeddings => {
        let scaled_embeddings = this.scaleEmbeddings(umap_embeddings)
        this.setState({
          umap_dims_0_2: scaled_embeddings,
        })
      })
    fetch(`${process.env.PUBLIC_URL}/umap_dims_0_1.json`)
      .then(response => response.json())
      .then(umap_embeddings => {
        let scaled_embeddings = this.scaleEmbeddings(umap_embeddings)
        console.log('got em')
        this.setState({
          umap_dims_0_1: scaled_embeddings,
        })
      })
    fetch(`${process.env.PUBLIC_URL}/umap_dims_1_2.json`)
      .then(response => response.json())
      .then(umap_embeddings => {
        let scaled_embeddings = this.scaleEmbeddings(umap_embeddings)
        this.setState({
          umap_dims_1_2: scaled_embeddings,
        })
      })
    fetch(`${process.env.PUBLIC_URL}/umap_train_war.json`)
      .then(response => response.json())
      .then(mnist_labels =>
        this.setState({
          mnist_labels: mnist_labels,
        })
      )
    fetch(`${process.env.PUBLIC_URL}/umap_train_dates.json`)
      .then(response => response.json())
      .then(mnist_dates =>
        this.setState({
          mnist_dates: mnist_dates,
        })
      )
  }

  render() {
    return this.state.umap_dims_0_2 && this.state.mnist_labels ? (
      <Layout
        {...this.state}
        algorithm_options={algorithm_options}
        algorithm_embedding_keys={algorithm_embedding_keys}
      />
    ) : (
        <div style={{ padding: '1rem' }}>Loading data...</div>
      )
  }
}

export default Data
