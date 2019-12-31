import React, { Component } from 'react'

class Sidebar extends Component {
  componentDidMount() {
    this.props.setSidebarCanvas(this.side_canvas)
    this.props.setSidebarCanvasAn1(this.side_canvas_an1)
    this.props.setSidebarCanvasAn2(this.side_canvas_an2)
    this.props.setSidebarCanvasAn3(this.side_canvas_an3)
    this.props.setSidebarCanvasAn4(this.side_canvas_an4)
    this.props.setSidebarCanvasAn5(this.side_canvas_an5)
    this.props.setSidebarCanvasAn6(this.side_canvas_an6)
    this.props.setSidebarCanvasAn7(this.side_canvas_an7)
    this.props.setSidebarCanvasAn8(this.side_canvas_an8)
    this.props.setSidebarCanvasAn9(this.side_canvas_an9)
    this.props.setSidebarCanvasAn10(this.side_canvas_an10)
    this.props.setSidebarCanvasAn11(this.side_canvas_an11)
    this.props.setSidebarCanvasAn12(this.side_canvas_an12)

    this.handleSelectAlgorithm = this.handleSelectAlgorithm.bind(this)
  }

  handleSelectAlgorithm(e) {
    let v = e.target.value
    this.props.selectAlgorithm(v)
  }

  render() {
    let {
      sidebar_orientation,
      sidebar_image_size,
      grem,
      p,
      hover_index,
      mnist_labels,
      mnist_dates,
      color_array,
      algorithm_options,
      algorithm_choice,
    } = this.props

    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          flexGrow: 1,
        }}
      >
        <div>
          {' '}
          {/* <div
            style={{
              padding: grem / 2,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div>Projection:</div>
            <select
              onChange={this.handleSelectAlgorithm}
              value={algorithm_options[algorithm_choice]}
            >
              {algorithm_options.map((option, index) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div> */}
          <div
            style={{
              display: 'flex',
              flexDirection:
                sidebar_orientation === 'horizontal' ? 'row' : 'column',
            }}
          >
            <div>
              <canvas
                ref={side_canvas => {
                  this.side_canvas = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an1 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an2 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an3 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an4 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an5 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an6 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an7 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an8 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an9 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an10 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an11 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
              <canvas
                ref={side_canvas => {
                  this.side_canvas_an12 = side_canvas
                }}
                width={sidebar_image_size}
                height={sidebar_image_size}
              />
            </div>
            <div style={{ flexGrow: 1 }}>
              <div
                style={{
                  padding: grem / 2,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div>Projection:</div>
                <select
                  onChange={this.handleSelectAlgorithm}
                  value={algorithm_options[algorithm_choice]}
                >
                  {algorithm_options.map((option, index) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
              <div
                style={{
                  background: hover_index
                    ? `rgb(${color_array[mnist_labels[hover_index]].join(',')})`
                    : 'transparent',
                  color: hover_index ? '#000' : '#fff',
                  padding: p(grem / 4, grem / 2),
                  display: 'flex',
                  justifyContent: 'space-between',
                  transition: 'all 0.1s linear',
                }}
              >
                <div>Wet Area Ratio:</div>
                {hover_index ? <div>{(mnist_labels[hover_index] / 255).toFixed(2)}</div> : null}
              </div>
              <div
                style={{
                  padding: p(grem / 4, grem / 2),
                  display: 'flex',
                  justifyContent: 'space-between',
                }}
              >
                Date:
                {hover_index ? <div>{mnist_dates[hover_index]}</div> : null}
              </div>
              <div
                style={{
                  padding: p(grem / 4, grem / 2),
                  display: 'flex',
                  justifyContent: 'space-between',
                }}
              >
                Index:
                {hover_index ? <div>{hover_index}</div> : null}
              </div>
            </div>
          </div>
        </div>
        <div style={{ padding: grem / 2 }}>
          <div>
            An interactive UMAP visualization of the TAASRAD19 dataset.{' '}
            <button
              onClick={() => {
                this.props.toggleAbout(true)
              }}
            >
              About
            </button>
          </div>
        </div>
      </div>
    )
  }
}

export default Sidebar
